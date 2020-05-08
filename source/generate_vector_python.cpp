#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <stdio.h>


#include "mcc_vector/mcc.h"


//arg1 lista de arquivos
//arg2 pasta de saida dos vetores
//arg3 parameter vector bin
//arg4 parameter vector bin
//arg5 quality

void find_hand_finger(std::string number_finger, int * finger, int * hand){
        int value = std::stoi(number_finger);
        if(value > 5){
                *finger = value - 6;
                *hand = 1;
        }
        else {
                *finger = value - 1 ;
                *hand = 0;
        }

}
void save_Finger_Person(Person * p , Finger * f, int finger, int hand, std::vector<std::vector< uint8_t>> vec){

        f->finger = finger;
        f->hand = hand;
        f->vec = vec;
        p->fing.push_back(*f);




}

std::vector<std::vector<uint8_t>> get_vector_bin(std::string filename, int R, int NS, int ND, int quali,std::string file,char * vector_p){


        MinuList ML;
        std::size_t found;

        int width, height;
        vector<vector<uint8_t>> bin_vector;

        if(!LoadMinutiaeList(filename.c_str(), ML,quali)){
                printf("error open the file .xyt\n");
                return bin_vector;
        }

        bin_vector = getMCCBinVector(ML, true, R, NS, ND);


        found = file.find_last_of(".");
        std::string vector_path = vector_p;

        vector_path += file.substr(0, found+1);
        vector_path += "txt";
        //std::cout << "AAAAAAAAAAAAAA " << vector_path << std::endl << std::endl << std::endl;



        FILE * saida = fopen(vector_path.c_str(), "w");
        if (!saida){
            printf("Open output file!\r\n");
            return bin_vector ;
        }
        
        for(size_t i = 0; i < bin_vector.size(); i++){
                for(size_t j = 0; j < bin_vector[i].size() - 1; j++){
                        fprintf(saida, "%d ",bin_vector[i][j]);

                   //     printf("%d ",bin_vector[i][j]);
                                              //  fprintf(saida, "%s\n",words.c_str());

                }
               // if(bin_vector[i].size() > 0){
                   //     cout << bin_vector[i][bin_vector[i].size() - 1] << "\n";
                    fprintf(saida,"\n");
           //         std::cout << "\n";
                //}
        }

        fclose(saida);
        return bin_vector;

}

int main(int argc, char * argv[]){

        std::size_t found,found2;
        std::string file, path, number_person, number_finger,filename,words,word_path;
        int inf;
        long int id;
        int sex, hand, finger; 
        std::vector<std::vector<uint8_t>> vec_bin;
        int count_finger = 0;
        int count_person = 0;
        FILE * fp;
        FILE * saida;
        char quality[2];
        printf("aaaaaaaaaaaaaa\n");




        FILE* search = fopen(argv[1], "r");
        if (!search){
                printf("Open index file!\r\n");
                return false;
        }

      //  fprintf(saida, "id,finger,hand,pos_rank,quality\n");


        int first = 1;
        char name[100];

        while(1){
                inf =  fscanf(search, "%s\n", name);
                filename = name;
                if(inf==EOF)
                {
                 
                        fclose(search);
                        break;
                }


                found = filename.find_last_of("/");
                path = filename.substr(0,found);
                file = filename.substr(found+1);
                found = path.find_last_of("/");
                number_person =  path.substr(found+1);

                found = file.find_last_of("-");
                found2 = file.find(".");
                number_finger =  file.substr(found+1 ,found2 - found - 1);

             //   if(!std::strcmp(number_finger.c_str(),"1")){
                        vec_bin = get_vector_bin(filename,70,atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),file,argv[2]);


                          

    //                    words = indexing.generateWord(vec_bin);
      //                  word_path = argv[2];
////
                    //    found = file.find_last_of(".");
                  //      word_path += file.substr(0, found+1);
                //        word_path += "txt";
              //          std::cout << "file " << file << " file_out "<< word_path << '\n';

          //              saida = fopen(word_path.c_str(), "w");
           //             if (saida){
             //               printf("Open output file!\r\n");
                     //       return false;
            //                fprintf(saida, "%s\n",words.c_str());
              //              count_person++;
                //            count_finger++;

                  //      }
                    //    fclose(saida);


                        
                //}       
        }

}
