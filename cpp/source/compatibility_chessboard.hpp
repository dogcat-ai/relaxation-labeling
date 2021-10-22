#ifndef _COMPATIBILITY_CHESSBOARD
#define _COMPATIBILITY_CHESSBOARD

#include "compatibility.hpp"

struct CompatibilityChessboard : Compatibility {
    CompatibilityChessboard (size_t numObjects, size_t numLabels);
    void defineObjectsAndLabels();
    void calculate();
    void calculate(FindChessboardCorners findChessboardCorners );
};

#endif// _COMPATIBILITY_CHESSBOARD

