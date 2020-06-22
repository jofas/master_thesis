MACHINE_VERTICES=$(wildcard make/*.mk)

all: $(MACHINE_VERTICES)
	$(foreach file,$^,$(MAKE) -f $(file);)

clean:
	rm build/*
	rm *.aplx
	rm *.dict
	rm -r modified_src
