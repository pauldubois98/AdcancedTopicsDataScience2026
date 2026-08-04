SLIDE_SOURCES := $(wildcard Course*/slides.md)
SLIDE_TARGETS := $(SLIDE_SOURCES:.md=.pptx)

.PHONY: slides clean-slides
slides: $(SLIDE_TARGETS)

%.pptx: %.md scripts/fit_pptx.py
	pandoc $< -o $@ --slide-level=2 --resource-path=$(dir $<)
	python3 scripts/fit_pptx.py $@

clean-slides:
	rm -f $(SLIDE_TARGETS)
