#!/bin/bash
PROJECT_NAME="comfyui-api"
echo "기존 $PROJECT_NAME 서버를 중지합니다."

docker stop $PROJECT_NAME ; docker rm $PROJECT_NAME ; docker rmi $PROJECT_NAME

# 빌드
docker build -t $PROJECT_NAME .

# 실행 (GPU + 모델 볼륨 마운트)
docker run -d \
  --name $PROJECT_NAME \
  --runtime=nvidia \
  --ipc=host \
  --restart=always \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  $PROJECT_NAME

docker logs $PROJECT_NAME --tail 20 -f
