#ifndef Person_H 
#ifndef Finger_H
#define Finger_H
#define Person_H

#include <iostream>
#include <stdio.h>
#include <vector>



class Finger{

	public:
		int hand;
		int finger;
		std::vector<std::vector<uint8_t > > vec;
		std::string words;
		std::string singular_points;
};

class Person{
	public:
		std::vector<Finger> fing;
		long int id;
		int sex;

};

#endif
#endif