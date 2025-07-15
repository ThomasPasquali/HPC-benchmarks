MPICXX = mpicxx

BASE_PATH = DNN-cpp-proxies/proxies/comm_only

all: gpt2 bert resnet

gpt2:
	$(MPICXX) $(BASE_PATH)/gpt2_large.cpp -o $@

resnet:
	$(MPICXX) $(BASE_PATH)/resnet50.cpp -o $@

bert:
	$(MPICXX) $(BASE_PATH)/bert_large.cpp -o $@

clean:
	rm gpt2 bert resnet