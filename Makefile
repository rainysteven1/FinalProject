PWD := $(shell pwd)

random-string:
	@echo $(shell openssl rand -hex 4)

run:
	@random_str=$(shell make random-string); \
	docker run -itd \
		--name sentiment-classification-$$random_str \
		--privileged \
		-v $(PWD):/app \
		-e TZ=Asia/Shanghai \
		project:latest python main.py

stop-and-remove-all:
	@docker stop $$(docker ps -q -f name=^sentiment-classification-); \
	docker rm $$(docker ps -aq -f name=^sentiment-classification-)

.PHONY: run random-string