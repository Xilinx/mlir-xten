#ifndef ARCH_H
#define ARCH_H

#include <stdint.h>
#include <math.h>

class AbsArchitecture {
 public:
    virtual ~AbsArchitecture() {};
    virtual uint64_t getBankSize() = 0;
    virtual uint64_t getNumBanks() = 0;
    virtual uint64_t getMemSize() = 0;
    virtual uint64_t getVectSize() = 0;
    virtual uint64_t getComSpeed() = 0;
    virtual uint64_t getPipelineDepth() = 0;
    virtual uint64_t getNumCores() = 0;
    virtual uint64_t getClockFrequency() = 0;
};

class AIEv1 : public AbsArchitecture {
 private:
    uint64_t xWidth;
    uint64_t zWidth;

 public:
 AIEv1(uint64_t acts, uint64_t weights) : xWidth(acts), zWidth(weights) {}
    ~AIEv1() {}

    // Size in bytes
    uint64_t getBankSize() {
        return pow(2, 12);
    }

    // Integer
    uint64_t getNumBanks() {
        return 8;
    }

    // Size in bytes
    uint64_t getMemSize() {
        return getBankSize() * getNumBanks();
    }

    // Integer
    uint64_t getVectSize() {
        //llvm::outs() << "Vects size is: " << 128 / (xWidth * zWidth) << "\n";
        return 128 / (xWidth * zWidth);
    }

    // Bytes per cycles
    uint64_t getComSpeed() {
        return 4;
    }

    // Integer, TODO check that
    uint64_t getPipelineDepth() {
        return 8;
    }

    uint64_t getNumCores() {
        return 400;
    }

    uint64_t getClockFrequency() {
        return pow(10, 9);
    }
};

#endif
