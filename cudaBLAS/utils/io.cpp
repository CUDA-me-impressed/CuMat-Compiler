#include <iostream>

struct HeaderI
{
	long* data;
	long rank;
	long bytes;
	long* dimensions;
};

struct HeaderD
{
	double* data;
	long rank;
	long bytes;
	long* dimensions;
};

extern "C" void printMatrixHI(HeaderI* h)
{
	if(h->rank == 1)
	{
		std::cout << "[";
		long len = h->dimensions[0];
		long* data = h->data;
		if (len > 0) {std::cout << data[0];}
		for(long l = 1; l < len; ++l)
		{
			std::cout << "," << data[l];
		}
		std::cout<<"]" << std::endl;
	} else if(h->rank == 2)
	{
		std::cout << "[";
		long width = h->dimensions[0];
		long height = h->dimensions[1];
		long* data = h->data;
		for(long row = 0; row < height; row++)
		{
			for(long col = 0; col < width; col++)
			{
				int loc = col + (row * width);
				std::cout << data[loc] << " ";
			}
			std::cout << "\\" << std::endl;
		}
		std::cout << "]" << std::endl;
	} else
	{
		std::cout << "Many dimensions is hard" << std::endl;
		std::cout << "[";
		long len = 0;
		for(long r = 0; r < h->rank; r++)
		{
			len += h->dimensions[r];
		}
		long* data = h->data;
		if (len > 0) {std::cout << data[0];}
		for(long l = 1; l < len; ++l)
		{
			std::cout << "," << data[l];
		}
		std::cout<<"]" << std::endl;
	}
}

extern "C" void printMatrixI(long* m, long len)
{
	std::cout << "[";
	if (len > 0) { std::cout << m[0]; }
	for(long l = 1; l < len; ++l)
	{
		std::cout << "," <<  m[l];
	}
	std::cout << "]" << std::endl;
}

extern "C" void printMatrixD(double* m, double len)
{
    std::cout << "[";
    if (len > 0) { std::cout << m[0]; }
    for(long l = 1; l < len; ++l)
    {
        std::cout << "," <<  m[l];
    }
    std::cout << "]" << std::endl;
}
