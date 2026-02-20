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
	-lopencv_highgui \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lopencv_videoio \
	-lopencv_objdetect


BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
UTILSDIR = ./src/utils


# Prerequisites / Dependencies
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/offline/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/online/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(UTILSDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Targets
all: fg matcher

COMMON_OBJS = $(OBJDIR)/csvUtil.o \
			  ${OBJDIR}/filters.o \
              $(OBJDIR)/readFiles.o \
			  $(OBJDIR)/extractorFactory.o \
			  $(OBJDIR)/extractor.o \

pretrain: $(OBJDIR)/preTrainer.o \
		 $(OBJDIR)/preTrainerCLI.o \
         $(COMMON_OBJS)
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)


clean:
	rm -rf obj/*.o bin/* *~ 
