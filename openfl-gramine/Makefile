ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

# This is a signer key on a BUILDING machine
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
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dno_proxy=$(no_proxy) \
		-Dhttp_proxy=$(http_proxy) \
		-Dhttps_proxy=$(https_proxy) \
		$< >$@

openfl.manifest.sgx: openfl.manifest
	@test -s $(SGX_SIGNER_KEY) || \
	    { echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
	gramine-sgx-sign \
		--key $(SGX_SIGNER_KEY) \
		--manifest $< \
		--output $@

openfl.sig: openfl.manifest.sgx

openfl.token: openfl.sig
	gramine-sgx-get-token --output $@ --sig $<


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT* *.PID TEST_STDOUT TEST_STDERR
	$(RM) -r scripts/__pycache__

.PHONY: distclean
distclean: clean