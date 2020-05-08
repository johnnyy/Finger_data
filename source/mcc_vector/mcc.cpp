#include "mcc.h"

// Extract MCC for one template
void ExtractMCC(MinuList& ML, int ND, int NS, int R, 
	float *realmcc, char* binmcc, bool Rotation)
{

	// initialization 
	MCC mcc;
	mcc.initialization(R, NS, ND);

	// Get local structure
	vector<MinuList> LocalML;
	ConstructLocalStructure(ML, LocalML, R + mcc.delta_S*3,Rotation);

	//

	for(int k=0;k<ML.size();k++)
	{
		mcc.ML = LocalML[k];
		mcc.CreateMCC(realmcc+mcc.MCC_Len*k, binmcc+mcc.MCC_Len*k);
	}
}

vector<vector<uint8_t>> getMCCBinVector(MinuList &ML, bool rotation, int R, int NS, int ND){
	vector<vector<uint8_t> > bin_vector;

	//int R = 70;
    //int NS = 16;
    //int ND = 5;
	MCC mcc;
	mcc.N_D=ND;
	mcc.N_S=NS;
	mcc.R=R;
	mcc.initialization(R, NS, ND);
	int Res = 500;

	int MCCLen = GetMCCLen(R, NS, ND);
    float	*realmcc = new float [MCCLen*ML.size()];
    char	*binmcc = new char [MCCLen*ML.size()];

    ExtractMCC(ML, ND, NS, R, realmcc, binmcc, rotation);

	bin_vector = SaveMccToVector(binmcc, MCCLen, ML.size(), mcc.msk_cell, NS, ND, ML);

	delete[] realmcc;
	delete[] binmcc;
	return bin_vector;
}

int GetMCCLen(float R, int NS, int ND)
{
	float D = 2*R/NS;
	int RT = 0;
	for(int i=1;i<=NS/2;i++)
	{
		for(int j=1;j<=NS/2;j++)
		{
			int x = (i-0.5)*D;
			int y = (j-0.5)*D;
			float dis = (x-R)*(x-R) + (y-R)*(y-R);
			if(dis<=R*R)
			{
				RT++;
			}
		}
	}
	return RT*ND*4;
}
bool LoadMCCMinutiaeList(const char* filename, MinuList& ML, int& width, int& height)
{
	FILE* f = fopen(filename, "r");
	if (!f)
	{
		printf("Open %s fail!\r\n", filename);
		return false;
	}

	ML.clear();

    int res = 0;
    int num = 0;
    fscanf(f, "%d", &width);
    fscanf(f, "%d", &height);
    fscanf(f, "%d", &res);
    fscanf(f, "%d", &num);

	for(int k=0;k<num;k++)
	{
		Minu M;
		int q;
		float theta;
		int e = fscanf(f, "%d %d %f", &M.x, &M.y, &theta);
        if(e==EOF)
        {
            fclose(f);
            return false;
        }
		M.theta = theta;
		ML.push_back(M);
	}

	fclose(f);
	return true;
}
bool LoadMinutiaeList(const char* filename, MinuList& ML, int threshold_quality)
{
	FILE* f = fopen(filename, "r");
	if (!f)
	{
		printf("Open %s fail!\r\n", filename);
		return false;
	}

	ML.clear();
	int count = 0;
	int total = 0;

	while(1)
	{
		Minu M;
		int q;
		int t;
		int e = fscanf(f, "%d %d %d %d", &M.x, &M.y, &t, &q);

		if(e==EOF)
		{
			break;
		}else{
			total++;
			if(q > threshold_quality){
				M.theta = 2*t * PI / 180.;
				ML.push_back(M);

			}
			else {
				count++;

			}

		}
	}
	printf("Porcentagem jodada fora: %f%%, total: %d total_jogado: %d \n",100.0*count/total,total,count);

	fclose(f);
	return true;
}


bool LoadSourceAfisMinutiaeList(const char* filename, MinuList& ML, int threshold_quality)
{
	

	std::ifstream file_finger(filename, std::ifstream::in);
	json jsonfinger;
	if(file_finger.is_open()){

		file_finger >> jsonfinger; 
		int pos = 0;
		while(!jsonfinger["minutiae"][pos].is_null()){
			Minu M;
			int q;
			M.x = jsonfinger["minutiae"][pos]["x"];
			M.y = jsonfinger["minutiae"][pos]["y"];
			M.theta = jsonfinger["minutiae"][pos]["direction"];
			ML.push_back(M);
			pos++;

		}
		file_finger.close();
	}

	else return false;

	
	return true;
}





// get the local minutia
void ConstructLocalStructure(MinuList &ML, vector<MinuList> &LocalML, float R,bool Rotation)
{
	LocalML.clear();
	LocalML.resize(ML.size());

	// find the local minutiae
	for(int i=0;i<ML.size();i++)
	{
		LocalML[i].push_back(ML[i]);

		for(int j=0;j<ML.size();j++)
		{
			if (i==j) continue;
			float d = (ML[i].x - ML[j].x)*(ML[i].x - ML[j].x) + (ML[i].y - ML[j].y)*(ML[i].y - ML[j].y);
			if (d<=R*R)
			{
				LocalML[i].push_back(ML[j]);
			}
		}
	}

	// change to local coordinate system
	for(int i=0;i<ML.size();i++)
	{
		for(int k=0;k<LocalML[i].size();k++)
		{
			float dx = LocalML[i][k].x - ML[i].x;
			float dy = LocalML[i][k].y - ML[i].y;
            if(Rotation) 
            {
                float x = dx * cos(ML[i].theta) - dy * sin(ML[i].theta);
                float y = -dx * sin(ML[i].theta) - dy * cos(ML[i].theta);
                LocalML[i][k].x = x+0.5;
                LocalML[i][k].y = y+0.5;
                LocalML[i][k].theta -= ML[i].theta;
		    }else{
                LocalML[i][k].x = dx;
                LocalML[i][k].y = dy;
            }
        }
	}
}

