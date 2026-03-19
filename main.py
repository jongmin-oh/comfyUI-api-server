import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import shutil
import folder_paths
import time
from comfy.cli_args import args, enables_dynamic_vram
from app.logger import setup_logger
import itertools
import utils.extra_config
import faulthandler
import logging
import sys
from comfy_execution.progress import get_progress_state

if __name__ == "__main__":
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)
faulthandler.enable(file=sys.stderr, all_threads=False)

import comfy_aimdo.control

if enables_dynamic_vram():
    comfy_aimdo.control.init()

if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'

if __name__ == "__main__":
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        os.environ['HIP_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc
    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ['OCL_SET_SVM_SIZE'] = '262144'


def apply_custom_paths():
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    if args.output_directory:
        folder_paths.set_output_directory(os.path.abspath(args.output_directory))

    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models", os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        folder_paths.set_input_directory(os.path.abspath(args.input_directory))

    if args.user_directory:
        folder_paths.set_user_directory(os.path.abspath(args.user_directory))


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    for custom_node_path in folder_paths.get_folder_paths("custom_nodes"):
        times = []
        for possible_module in os.listdir(custom_node_path):
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or possible_module == "__pycache__":
                continue
            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    continue
                t = time.perf_counter()
                ok = execute_script(script_path)
                times.append((time.perf_counter() - t, module_path, ok))
        if times:
            logging.info("\nPrestartup times for custom nodes:")
            for n in sorted(times):
                logging.info("{:6.1f} seconds{}: {}".format(n[0], "" if n[2] else " (FAILED)", n[1]))


apply_custom_paths()
execute_prestartup_script()


# ---------------------------------------------------------------------------
import asyncio
import threading
import gc

if 'torch' in sys.modules:
    logging.warning("WARNING: Torch already imported before this point.")

import comfy.utils
import execution
import server
import nodes
import comfy.model_management
import app.logger
import hook_breaker_ac10a0
import comfy.memory_management
import comfy.model_patcher

if args.enable_dynamic_vram or (enables_dynamic_vram() and comfy.model_management.is_nvidia() and not comfy.model_management.is_wsl()):
    if (not args.enable_dynamic_vram) and (comfy.model_management.torch_version_numeric < (2, 8)):
        logging.warning("DynamicVRAM requires Pytorch 2.8+. Falling back to legacy ModelPatcher.")
    elif comfy_aimdo.control.init_device(comfy.model_management.get_torch_device().index):
        if args.verbose == 'DEBUG':
            comfy_aimdo.control.set_log_debug()
        elif args.verbose == 'CRITICAL':
            comfy_aimdo.control.set_log_critical()
        elif args.verbose == 'ERROR':
            comfy_aimdo.control.set_log_error()
        elif args.verbose == 'WARNING':
            comfy_aimdo.control.set_log_warning()
        else:
            comfy_aimdo.control.set_log_info()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
        logging.info("DynamicVRAM support detected and enabled")
    else:
        logging.warning("No working comfy-aimdo install detected. DynamicVRAM disabled.")


def cuda_malloc_warning():
    device_name = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                logging.warning("WARNING: this card may not support cuda-malloc. Run with --disable-cuda-malloc if you get CUDA errors.")
                break


def prompt_worker(q, server_instance):
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_args={"lru": args.cache_lru, "ram": args.cache_ram})
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0
    current_time = 0.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            start = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            extra_data = item[3].copy()
            for k in item[5]:
                extra_data[k] = item[5][k]

            e.execute(item[2], prompt_id, extra_data, item[4])
            need_gc = True

            q.task_done(item_id, e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages),
                        process_item=lambda p: p[:5] + p[6:])

            current_time = time.perf_counter()
            elapsed = current_time - start
            if elapsed > 600:
                logging.info("Prompt executed in {}".format(time.strftime('%H:%M:%S', time.gmtime(elapsed))))
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(elapsed))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()


def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        if node_id is None and ctx is not None:
            node_id = ctx.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if node_id is not None:
            get_progress_state().update_progress(node_id, value, total, preview_image)

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import cuda_malloc

    if sys.version_info < (3, 10):
        logging.warning("Python < 3.10 detected. 3.12+ recommended.")

    if args.temp_directory:
        folder_paths.set_temp_directory(os.path.join(os.path.abspath(args.temp_directory), "temp"))
    cleanup_temp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    prompt_server = server.PromptServer(loop)

    hook_breaker_ac10a0.save_functions()
    loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes,
    ))
    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    hijack_progress(prompt_server)

    threading.Thread(target=prompt_worker, daemon=True, args=(prompt_server.prompt_queue, prompt_server)).start()

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)

    from sdapi.executor import set_server as sdapi_set_server
    sdapi_set_server(prompt_server)

    from sdapi import app as sdapi_app
    import uvicorn

    logging.info("Starting SDAPI server on 0.0.0.0:7860")
    app.logger.print_startup_warnings()

    config = uvicorn.Config(sdapi_app, host="0.0.0.0", port=7860, loop="none", log_level="info")
    uvicorn_server = uvicorn.Server(config)

    try:
        loop.run_until_complete(uvicorn_server.serve())
    except KeyboardInterrupt:
        logging.info("\nStopped server")
    finally:
        cleanup_temp()
