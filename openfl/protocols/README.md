# Compile Protocol Buffers description files

After changing proto files run 
```
./compile_proto.sh
```
to recompile. It will delete old generated python 
files (*_pb2.py and *_pb2_grpc.py) and generate new.
Generated files should be committed with proto files.