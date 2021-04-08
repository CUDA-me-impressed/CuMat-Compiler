#include <iostream>

void printMatrix(long* m, long len)
{
	std::cout << "[";
	if (len > 0) { std::cout << m[0]; }
	for(long l = 1; l < len; ++l)
	{
		std::cout << "," <<  m[l];
	}
	std::cout << "]" << std::endl;
}
