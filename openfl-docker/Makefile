build:
	@docker build --pull -t openfl -f Dockerfile.base ..
run:
	@docker run --rm -it --network host openfl
save:
	@docker save openfl | gzip > openfl.tar.gz
