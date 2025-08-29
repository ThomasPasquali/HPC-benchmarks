MPICXX = mpicxx

BASE_PATH = DNN-cpp-proxies/proxies

all: gpt2 bert resnet resnet152

# Communication only
gpt2:
	$(MPICXX) $(BASE_PATH)/comm_only/gpt2_large.cpp -o $@

resnet:
	$(MPICXX) $(BASE_PATH)/comm_only/resnet50.cpp -o $@

bert:
	$(MPICXX) $(BASE_PATH)/comm_only/bert_large.cpp -o $@

# Emulated computed (using sleep)
resnet152:
	$(MPICXX) $(BASE_PATH)/resnet152.cpp -o $@

gpt3:
	$(MPICXX) $(BASE_PATH)/gpt3.cpp -o $@

clean:
	rm gpt2 bert resnet