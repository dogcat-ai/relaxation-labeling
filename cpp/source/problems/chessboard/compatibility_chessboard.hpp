#ifndef _COMPATIBILITY_CHESSBOARD
#define _COMPATIBILITY_CHESSBOARD

#include "../compatibility.hpp"

struct CompatibilityChessboard : Compatibility3Pairs {
    enum class Direction {UP, DOWN, LEFT, RIGHT};

    CompatibilityChessboard (size_t numObjects, size_t numLabels);
    void defineObjectsAndLabels();
    void calculate();
    void calculate(FindChessboardCorners findChessboardCorners );
    void calculateOneCorner(size_t r,size_t c,size_t numRows,size_t numColumns);
    double calculateOneDirection(Direction direction, size_t rowOrigin, size_t columnOrigin, size_t rowNear, size_t columnNear, size_t rowFar, size_t columnFar);

};

#endif// _COMPATIBILITY_CHESSBOARD

