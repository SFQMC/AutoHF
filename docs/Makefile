# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Create a symbolic link to the examples directory
examples-link:
	@echo "Creating symbolic link to examples directory..."
	@rm -f $(SOURCEDIR)/user/examples
	@ln -sf $(SOURCEDIR)/../../examples $(SOURCEDIR)/user/examples

# Add linkcheck dependency on examples-link to ensure the link exists before checking
linkcheck: examples-link

html: examples-link

.PHONY: help Makefile examples-link

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
