SLIDE_SOURCES := $(wildcard Course*/slides.md)
SLIDE_TARGETS := $(SLIDE_SOURCES:.md=.pptx)

.PHONY: slides clean-slides
slides: $(SLIDE_TARGETS)

%.pptx: %.md
	pandoc $< -o $@ --slide-level=2

clean-slides:
	rm -f $(SLIDE_TARGETS)
