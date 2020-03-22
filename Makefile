NAME?=fca_notebook

.PHONY: all build stop run logs

all: stop build run logs

build:
	docker build \
	-t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

logs:
	docker logs -f $(NAME)

run:
	docker run --rm -it \
		--net=host \
		--ipc=host \
		--name=$(NAME) \
		-v "$(PWD):/opt" \
		$(NAME)
