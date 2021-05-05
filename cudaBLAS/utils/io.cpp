#include <iostream>
#include <fstream>
#include <string>
#include "headers.hpp"
#include <exception>

std::string printMatrixIRankRec(long rank, long offset, HeaderI* h)
{
    if(rank < 0)
    {
        throw new std::runtime_error("Tried to print a negative rank matrix! : " + std::to_string(rank));
    }

    if(rank == 0) //Empty
    {
        return "";
    }

    std::string output;

    if(rank == 1) //Return row
    {
        long rowLength = h->dimensions[0];
        auto data = h->data;
        if(rowLength > 0) {output += std::to_string(data[0 + offset]);}
        for(long l = 1; l < rowLength; ++l)
        {
            output += ", " + std::to_string(data[l+offset]);
        }
        return output;
    }

    long rankLengthBelow = h->dimensions[rank - 2]; //As rank must be 2 or more here, this is safe
    long dimLength = h->dimensions[rank - 1];
    if(dimLength > 0)
    {
        output += printMatrixIRankRec(rank - 1, offset, h);
    }
    for(long l = 1; l < dimLength; ++l)
    {
        long newOffset = offset + l*rankLengthBelow;
        std::string dimSep;
        for(int i = 0; i < rank - 1; i++) //rank 2 matrix should have a single \ for separation
        {
            dimSep += "\\"; //One backslash, escaped
        }
        output += " " + dimSep + "\n" + printMatrixIRankRec(rank - 1, newOffset, h);
    }
    return output;
}

std::string printMatrixDRankRec(long rank, long offset, HeaderD* h)
{
    if(rank < 0)
    {
        throw new std::runtime_error("Tried to print a negative rank matrix! : " + std::to_string(rank));
    }

    if(rank == 0) //Empty
    {
        return "";
    }

    std::string output;

    if(rank == 1) //Return row
    {
        long rowLength = h->dimensions[0];
        auto data = h->data;
        if(rowLength > 0) {output += std::to_string(data[0 + offset]);}
        for(long l = 1; l < rowLength; ++l)
        {
            output += ", " + std::to_string(data[l+offset]);
        }
        return output;
    }

    long rankLengthBelow = h->dimensions[rank - 2]; //As rank must be 2 or more here, this is safe
    long dimLength = h->dimensions[rank - 1];
    if(dimLength > 0)
    {
        output += printMatrixDRankRec(rank - 1, offset, h);
    }
    for(long l = 1; l < dimLength; ++l)
    {
        long newOffset = offset + l*rankLengthBelow;
        std::string dimSep;
        for(int i = 0; i < rank - 1; i++) //rank 2 matrix should have a single \ for separation
        {
            dimSep += "\\"; //One backslash, escaped
        }
        output += " " + dimSep + "\n" + printMatrixDRankRec(rank - 1, newOffset, h);
    }
    return output;
}

extern "C" void printMatrixI(HeaderI* h)
{
    std::string output;

    output += printMatrixIRankRec(h->rank,0,h);

    output = "[\n" + output + "\n]";
    std::cout << output << std::endl;
    return;
}


extern "C" void printMatrixD(HeaderD* h)
{
    std::string output;

    output += printMatrixDRankRec(h->rank,0,h);

    output = "[\n" + output + "\n]";
    std::cout << output << std::endl;
    return;
}

std::vector<std::string> readFile(const std::string& fileName)
{
    std::ifstream fileStream = std::ifstream(fileName);

    if(!fileStream.is_open())
    {
        throw std::runtime_error("Could not find file: " + fileName);
    }

    std::vector<std::string> fileLines;
    std::string line;
    while (std::getline(fileStream, line)) fileLines.push_back(line);

    fileStream.close();
}

extern "C" HeaderI * readFromFileI(const char* file, long* dims, long rank)
{
    std::string fileName = std::string(file);
    auto fileContents = readFile(fileName);
    std::ostringstream concatenatedLines;
    for (auto line : fileContents)
    {
        concatenatedLines << line;
    }
    std::string contents = concatenatedLines.str();

    auto matHeader = new HeaderI;
    matHeader->rank = rank;
    //Will have to create in CodeGen
    matHeader->dimensions = dims;

    long totalElements = dims[0];
    for (long i = 1; i < rank; ++i)
    {
        totalElements *= dims[i];
    }

    matHeader->length = totalElements * sizeof(long);

    std::istringstream intConsumer = std::istringstream(contents);

    auto data = new long[totalElements];

    for (long i = 0; i < totalElements; ++i)
    {
        long x;
        char comma;
        intConsumer >> x >> comma;
        data[i] = x;
    }

    matHeader->data = data;

    return matHeader;
}

extern "C" HeaderD* readFromFileD(char* file, long* dims, long rank)
{
    std::string fileName = std::string(file);
    auto fileContents = readFile(fileName);
    std::ostringstream concatenatedLines;
    for (auto line : fileContents)
    {
        concatenatedLines << line;
    }
    std::string contents = concatenatedLines.str();

    auto matHeader = new HeaderD;
    matHeader->rank = rank;
    //Will have to create in CodeGen
    matHeader->dimensions = dims;

    long totalElements = dims[0];
    for (long i = 1; i < rank; ++i)
    {
        totalElements *= dims[i];
    }

    matHeader->length = totalElements * sizeof(double);

    std::istringstream intConsumer = std::istringstream(contents);

    auto data = new double[totalElements];

    for (long i = 0; i < totalElements; ++i)
    {
        double x;
        char comma;
        intConsumer >> x >> comma;
        data[i] = x;
    }

    matHeader->data = data;

    return matHeader;
}