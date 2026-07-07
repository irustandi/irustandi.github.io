HUGO ?= hugo
HUGO_OPTS ?=

.PHONY: help html clean serve publish

help:
	@echo 'Makefile for the Hugo site'
	@echo
	@echo 'Usage:'
	@echo '   make html              build the site into public/'
	@echo '   make publish           build the production site'
	@echo '   make serve [PORT=1313] serve locally with drafts and future posts'
	@echo '   make clean             remove generated Hugo output'

html:
	$(HUGO) $(HUGO_OPTS)

publish:
	HUGO_ENVIRONMENT=production HUGO_ENV=production $(HUGO) --gc --minify $(HUGO_OPTS)

serve:
ifdef PORT
	$(HUGO) server --buildDrafts --buildFuture --port $(PORT) $(HUGO_OPTS)
else
	$(HUGO) server --buildDrafts --buildFuture $(HUGO_OPTS)
endif

clean:
	rm -rf public resources .hugo_build.lock
