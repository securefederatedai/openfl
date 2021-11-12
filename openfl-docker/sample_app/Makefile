ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
# ARCH_LIBDIR ?= /lib/x86_64-linux-gnu

# This is a signer key on a BUILDING machine
# In this case we have a key inside the container
SGX_SIGNER_KEY ?= /var/gramine/Pal/src/host/Linux-SGX/signer/enclave-key.pem
PYTHON ?= $(shell sh -c "command -v python3")

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: sample_app.manifest
ifeq ($(SGX),1)
all: sample_app.manifest.sgx sample_app.sig sample_app.token
endif

# gramine-manifest -Dlog_level=debug -Darch_libdir=/lib/x86_64-linux-gnu -Dentrypoint=/usr/bin/python3.8 sample_app.manifest.template >sample_app.manifest
# python3.8 /usr/local/bin/gramine-manifest -Dlog_level=debug -Darch_libdir=/lib/x86_64-linux-gnu -Dentrypoint=/usr/bin/python3.8 sample_app.manifest.template >sample_app.manifest
sample_app.manifest: sample_app.manifest.template
	$(PYTHON) $(shell sh -c "command -v gramine-manifest") \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(shell sh -c "command -v $(PYTHON)") \
		$< >$@

# gramine-sgx-sign --key /gramine/Pal/src/host/Linux-SGX/signer/enclave-key.pem --manifest sample_app.manifest --output sample_app.manifest.sgx
sample_app.manifest.sgx: sample_app.manifest
	@test -s $(SGX_SIGNER_KEY) || \
	    { echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
	gramine-sgx-sign \
		--key $(SGX_SIGNER_KEY) \
		--manifest $< \
		--output $@

sample_app.sig: sample_app.manifest.sgx

# gramine-sgx-get-token --output sample_app.token --sig sample_app.sig
sample_app.token: sample_app.sig
	gramine-sgx-get-token --output $@ --sig $<

# gramine-sgx ./sample_app main.py bar
.PHONY: run
run: all
	gramine-sgx ./sample_app main.py bar

.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT* *.PID TEST_STDOUT TEST_STDERR
	$(RM) -r scripts/__pycache__

.PHONY: distclean
distclean: clean