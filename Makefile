# make without explicit target should build executables
.DEFAULT_GOAL := all

# OSX compiler
#CC = clang++

# Dwarf compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

CXX = $(CC)

# OSX include paths (for MacPorts)
#CFLAGS = -I/opt/local/include -I../include

MACOS_VERSION = 26.2

# OSX include paths (for homebrew, probably)
CFLAGS = -Wc++17-extensions \
	-std=c++17 \
	-O3 \
	-mmacosx-version-min=$(MACOS_VERSION) \
	-I/opt/homebrew/include/opencv4 \
	-I./include \
	-DENABLE_PRECOMPILED_HEADERS=OFF

# Dwarf include paths
#CFLAGS = -I../include # opencv includes are in /usr/include
CXXFLAGS = $(CFLAGS)

# OSX Library paths (if you use MacPorts)
#LDFLAGS = -L/opt/local/lib

#OSX Library paths (if you use homebrew, probably)
#LDFLAGS = -L/usr/local/lib

# Dwarf Library paths

LDFLAGS = -L/opt/homebrew/opt/opencv/lib -mmacosx-version-min=$(MACOS_VERSION)
# LDFLAGS = -L/opt/local/lib/opencv4/3rdparty -L/opt/local/lib # opencv libraries are here

# opencv libraries
# LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lIlmImf -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect
LDLIBS = \
	-lopencv_core \
	-lopencv_dnn \
	-lopencv_highgui \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lopencv_videoio \
	-lopencv_objdetect

# Optional ONNX Runtime integration:
# Example:
#   make ONNXRUNTIME_DIR=/opt/homebrew/opt/onnxruntime
ONNXRUNTIME_DIR ?= $(shell if [ -d /opt/homebrew/opt/onnxruntime/include ] && [ -d /opt/homebrew/opt/onnxruntime/lib ]; then echo /opt/homebrew/opt/onnxruntime; fi)
ifneq ($(ONNXRUNTIME_DIR),)
	CXXFLAGS += -DENABLE_ONNXRUNTIME -I$(ONNXRUNTIME_DIR)/include
	LDFLAGS += -L$(ONNXRUNTIME_DIR)/lib
	LDLIBS += -lonnxruntime
endif


BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
DATADIR = ./data
UTILSDIR = ./src/utils

# --- directory targets ---
$(OBJDIR):
	mkdir -p $@

$(BINDIR):
	mkdir -p $@

$(DATADIR):
	mkdir -p $@

# --- build object files (order-only: ensure obj dir exists) ---
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/offline/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/online/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(UTILSDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- targets ---
all: pretrain rtor

COMMON_OBJS = $(OBJDIR)/csvUtil.o \
			  $(OBJDIR)/extractorFactory.o \
			  $(OBJDIR)/extractor.o \
			  ${OBJDIR}/filters.o \
			  $(OBJDIR)/preProcessor.o \
			  $(OBJDIR)/regionAnalyzer.o \
              $(OBJDIR)/readFiles.o \
			  $(OBJDIR)/regionDetect.o \
			  $(OBJDIR)/utilities.o \
			  $(OBJDIR)/thresholding.o \
			  $(OBJDIR)/morphologicalFilter.o

pretrain: $(OBJDIR)/preTrainer.o \
          $(OBJDIR)/preTrainerCLI.o \
          $(COMMON_OBJS) \
		  | $(BINDIR) $(DATADIR)
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

rtor: $(OBJDIR)/RTObjectRecognitionApp.o \
	  $(OBJDIR)/distanceMetrics.o \
	  $(OBJDIR)/featureMatcher.o \
	  $(OBJDIR)/main.o \
	  $(OBJDIR)/matchUtil.o \
	  $(OBJDIR)/metricFactory.o \
      $(COMMON_OBJS) \
      | $(BINDIR) $(DATADIR)
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf obj/*.o bin/* *~ 
