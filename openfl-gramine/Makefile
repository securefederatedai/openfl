ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

# Enclave attributes to the manifest
SGX_ISVPRODID ?= 0
SGX_ISVSVN ?= 0
SGX_ENCLAVE_SIZE ?= 16G

# This is a signer key on the BUILDING machine
SGX_SIGNER_KEY ?= /key.pem

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: openfl.manifest
ifeq ($(SGX),1)
all: openfl.manifest.sgx openfl.sig openfl.token
endif

openfl.manifest: openfl.manifest.template
	@echo "Making openfl.manifest file"
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Disvprodid=$(SGX_ISVPRODID) \
		-Disvsvn=$(SGX_ISVSVN) \
		-Denclave_size=$(SGX_ENCLAVE_SIZE) \
		-Dno_proxy=$(no_proxy) \
		-Dhttp_proxy=$(http_proxy) \
		-Dhttps_proxy=$(https_proxy) \
		$< >$@

openfl.manifest.sgx: openfl.manifest
	@echo "Making openfl.manifest.sgx file"
	@test -s $(SGX_SIGNER_KEY) || \
	    { echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
	gramine-sgx-sign \
		--key $(SGX_SIGNER_KEY) \
		--manifest $< \
		--output $@

openfl.sig: openfl.manifest.sgx

openfl.token: openfl.sig
	@echo "Making openfl.sig file"
	gramine-sgx-get-token --output $@ --sig $<


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT* *.PID TEST_STDOUT TEST_STDERR
	$(RM) -r scripts/__pycache__

.PHONY: distclean
distclean: clean