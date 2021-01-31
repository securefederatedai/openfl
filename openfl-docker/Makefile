build:
	@docker build -t openfl -f Dockerfile.base ..
run:
	@docker run -it --network host openfl
save:
	@docker save openfl > openfl.tar