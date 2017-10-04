
#include <stdio.h>
#include <math.h>
//#include "L138_LCDK_aic3106_init.h"
//#include "L138_LCDK_switch_led.h"
#include "L138_LCDK_aic3106_init.h"
#include "evmomapl138_gpio.h"
#include "fft.h"

/********************************************************************
 ************* Adjustable Parameters **********
 *******************************************************************/
#define frame_size 256 //single frame size (32ms)
#define frame_num 32
#define SAMPLING_f 8000 //remember to change the value in the initiation process
#define THRESHOLD 1000000 //Sensitivity of the microphone
#define data_base_size 5 //size of the library

/********************************************************************
 ************* Constant Parameters **********
 *******************************************************************/
#define filter_num 26
#define MFCC_size 13
#define full_size frame_num*frame_size //valid signal sze (0.512s*2)
#define a 0.54
#define b 0.46
#define K full_size/2+1
#define upper_f 8000.0
#define lower_f 250.0
#define avgk 8000.0/513.0 //delta f in k domain
#define PI 3.14159265358979

/********************************************************************
 ************* General variables used in loops and debug **********
 *******************************************************************/

int n = 0; //general iterator
int j = 0; //general iterator
int m=0; //filter number iterator
int k=0; //filter bank iterator
int debug_count = 0;

/********************************************************************
 ************* Variables and Arrays for sample collections **********
 *******************************************************************/

int16_t frame[full_size]; //the array that contain the entire valid signal input (0.512s)
#pragma DATA_SECTION(frame, ".EXT_RAM")

int16_t frame_seg[frame_size]; //32ms frame segment
int16_t scout_squad[3];

int current = 0; //record how many invalid data has passed
int check = 0;
int iterator = 0; //iterator used in interrupt so dont reuse it
int inst = 0; //type-in command  0:no command 1:start recording 2:reboot
int i=0; //sample
int counter = 0; //time counter
int signal = 0; //signal to start recording

/********************************************************************
 ************* Variables and Arrays for feature extraction **********
 *******************************************************************/

COMPLEX twiddle[frame_size];
COMPLEX samples[frame_size];

float w[full_size];  //Hamming window function array
#pragma DATA_SECTION(w,".EXT_RAM")

float periodogram [K]; //store the outcome of s[n]*w[n]
#pragma DATA_SECTION(periodogram,".EXT_RAM")

float p_mag[frame_size]; // store the magnitude of N_FFT
#pragma DATA_SECTION(p_mag,".EXT_RAM")

float p_mag_sqr[frame_size]; // store the values of FFT with square and division by N
#pragma DATA_SECTION(p_mag_sqr,".EXT_RAM")

float PSD[K]; // take first K values of p_mag_sqr array to PSD to generate first K values.
#pragma DATA_SECTION(PSD,".EXT_RAM")

float mel_fre[28];  //mel frequency (26 filters -> 28 frequency bin)
float real_fre[28];  //original friquency (26 filters -> 28 frequency bin)
float avg=0; //the equal difference between adjacent mel-frequency
float energy[filter_num]; //sum of the product at each filter
float log_energy[filter_num]; //10 th log of the energy array
float MFCC[MFCC_size];  //MFCC vector values
float MFCC_TOTAL[frame_num][MFCC_size]; //MFCC for the entire input signal
float Eucliean_d[4]; //Eucliean distance between the unknown vowel to the known vowel and vowel recongnition
float min_Eucliean=0; //record the minimum Eucliean distance
float Eucliean_d_sqr[4];

int vowel_index = 0; //indicate where the minimum Eucliean index locates
int memoryLoad = 0;

/********************************************************************
 ************* Variables and Arrays for DTW **********
 *******************************************************************/

float dist[data_base_size][frame_num][MFCC_size][MFCC_size];
#pragma DATA_SECTION(dist,".EXT_RAM")

float DTW[data_base_size][frame_num][MFCC_size][MFCC_size];
#pragma DATA_SECTION(DTW,".EXT_RAM")

// Resulting  of least costly (distance) between MFCC values
// Contains indices of those match ups
int path[data_base_size][frame_num][MFCC_size*2][2];
#pragma DATA_SECTION(path,".EXT_RAM")

// Records the lengths of the paths per window per character
int path_lengths_records[data_base_size][frame_num];
#pragma DATA_SECTION(path_lengths_records,".EXT_RAM")

// Records the path_cost for each character
// used to determine what character is detected by the minimum path_cost
float path_cost[data_base_size];
#pragma DATA_SECTION(path_cost,".EXT_RAM")

// MISC calculation arrays
float DTW_local[5];

//delta and delta delta arrays
float d_array[MFCC_size];
float d_d_array[MFCC_size];

int pathx;
int path_length;
int min_index;
int x; //x axis of the DTW matrix
int y; //y axis of the DTw matrix

float distance; //DTW distance
float DTW_min;  //the smallest DTW differences

// delta index
float delta;
int idx1 = 0;
int idx2 = 0;

int p = 0; //general iterator

/********************************************************************
 ************* Variables and Arrays for Data base construction **********
 *******************************************************************/

struct Data {
	char *vocab;
	char *name;
	float MFCC_info[frame_num][MFCC_size];
};

struct Data Library[data_base_size];  //the library we are going to build contains								  //data_base size number of vocab
char *file_name[data_base_size];     //store all file names in this array

/********************************************************************
 ************* Variables and Arrays for Machine Learning **********
 *******************************************************************/

char judgement; //human judgment of the vowel
char correction; //human correction

/********************************************************************
 ************* Synchronization Functions **********
 *******************************************************************/

int g = 0; //signal check iterator iterator

int signal_check(); //detect the start of a signal
void syc_initiation();  //initialize arrays and parameter used in synchronization
void sychronization();  //synchronization interrupt
void sychronization_process(); //synchronization main

/********************************************************************
 ************* Feature extraction Functions **********
 *******************************************************************/

int frame_iterator = 0;

void MFCC_extraction_process();
void Partial_initialization(); //initialize some of arrays used in extraction process
void Hamming_window ();
void Periodogram();
void FFT();
void FFT_square();
void power_spectral();
void mel_f();
void real_f();
void filterbank();
void filerPSD();
void MFCC_transform();

/********************************************************************
 ************* Library Construction Functions **********
 *******************************************************************/

int free_space = data_base_size;  //used to choose to continue or exit library construction

void library_initiation();
void data_base_construction(int num, char* word);
void clear_library();
void memory_write(int lib_num);
void memory_load();
void file_name_load();

/********************************************************************
 ************* Other Functions **********
 *******************************************************************/

void initialization();
void reboot_inquiry();
//void machine_learning(int sw);

/********************************************************************
 ************* DTW Functions **********
 *******************************************************************/

float min_func(float x, float y, float z);

void delta_f();
void delta_delta();
void least_cost();
void DTW_process();

interrupt void interrupt4(void) // interrupt service routine
{
	sychronization();
	return;
}

int main(void)

{
  //LCDK_GPIO_init();
  //LCDK_SWITCH_init();
  //LCDK_LED_init();

  while (1)
  {
	if (signal == 0) //if the interrupt process has not been initiated
	{
		printf("*********VOICE RECOGNITION MACHINA CEREAL VERSION 1.0**************\n");
		printf("Command Option:\n");
		printf("1. Voice recognition \n2. Load memory \n3. Library construction\n4. Clear the Library\n");
		printf("There are %d free slot left in the library. \n", free_space);
		printf("Please type in your command, my majesty.\n");
		scanf("%d", &inst);

		if (inst == 1 || inst == 3)
		{
			if (inst == 1 && free_space == data_base_size)
			{
				printf("Command not allowed! There is nothing to compare with! \n");
				printf("Extraction process fails!\n");
			}
			else
			{
				signal = 1; //tell the interrupt to start recording

				if (inst == 1) //tell the user they are in the voice recognition mode
					printf("Voice recognition process initiated...\n")

				if (inst == 3) //tell the user they are in the library construction mode
					printf("Library construction process initiated...\n");

				printf("Voice analysis process initiated!\n");
				printf("Waiting for Voice input...\n");
				initialization(); 			//initialize the array
				syc_initiation();
				L138_initialise_intr(FS_8000_HZ, ADC_GAIN_24DB, DAC_ATTEN_0DB, LCDK_MIC_INPUT);

				if (iterator == full_size && memoryLoad != 1) // if it did not detect the start of the input, i wont't increment to 1024
				{
					printf("get it! Processing...\n");
					MFCC_extraction_process();
					printf("process finish. \n");

					if (inst == 3) //store the MFCC data into the library
					{
						int data_index = 0;  //specify which vocab to build
						char* temp_word = NULL ;
						temp_word = (char *)malloc(8*sizeof(char));
						printf("Which Library slot you want to save? (1,2,3,4,5) \n");
						scanf("%d",&data_index);
						printf("Which word you just said? \n");
						scanf("%s",&temp_word[0]); // may need a space
						file_name_load(); //load in all file names
						data_base_construction(data_index, temp_word);  //standard vowels data_base construction
						memory_write(data_index);
					}

					if (inst == 1) //start the word matching process
					{
						//put the DTW and compare algorithm here
					}
				}
			}
		 }

		if (inst == 2) //memory loading
			{
				printf("Load memory? (N/Y)\n");
				char temp_ch = '\0';
				scanf(" %c",&temp_ch);
				printf ("\n");
				if (temp_ch == 'y' || temp_ch == 'Y')
				{
					memory_load();
					memoryLoad = 1;
				}
			}

		if (inst == 4)
		{
			clear_library();
		}

		reboot_inquiry(); //resume to menu
	}
}

void initialization()
{
	for (j=0 ; j<frame_size; j++)
	{
		frame_seg[j]=0;
		w[j]=0;
		periodogram[j]=0;
		p_mag[j]=0;
		p_mag_sqr[j]=0;
	}

	for (j=0 ; j<28 ; j++)
	{
		mel_fre[j]=0;
		real_fre[j]=0;
	}

	for (j=0; j<K; j++)
	{
		PSD[j]=0;
	}

	for (j=0; j<filter_num; j++)
	{
		energy[j]=0;
		log_energy[j]=0;
	}

	for (n=0; n<frame_num; n++) //initialize every MFCC values for the entire input signal
	{
		for (j=0; j<MFCC_size; j++)
		{
			if (n==0)  //initialize once is enough
				MFCC[j]=0;

			MFCC_TOTAL[n][j]=MFCC[j];
		}
	}

	for (j=0; j<4; j++)
	{
		Eucliean_d[j]=0;
		Eucliean_d_sqr[j]=0;
	}

	for (j=0 ; j<frame_size ; j++) //set up FFT twiddle factors
    {
 	    twiddle[j].real = cos(PI*j/frame_size);
	    twiddle[j].imag = -sin(PI*j/frame_size);
    }
}

void Partial_initialization()
{
	for (j=0 ; j<frame_size; j++)
	{
		frame_seg[j]=0;
		w[j]=0;
		periodogram[j]=0;
		p_mag[j]=0;
		p_mag_sqr[j]=0;
	}

	for (j=0 ; j<28 ; j++)
	{
		mel_fre[j]=0;
		real_fre[j]=0;
	}

	for (j=0; j<K; j++)
	{
		PSD[j]=0;
	}

	for (j=0; j<filter_num; j++)
	{
		energy[j]=0;
		log_energy[j]=0;
	}

	for (j=0; j<MFCC_size; j++)
	{
		MFCC[j]=0;
	}
}

void Hamming_window ()
{
	for (n=0;n<frame_size;n++)
		w[n]=a-b*cos(2*PI*n/(frame_size-1));
}

void Periodogram ()
{
	for (n=0;n<frame_size;n++)
		periodogram[n]=frame_seg[n]*w[n];
}

void FFT()
{
		  for (j=0 ; j<frame_size ; j++) //set up the real values of samples to be the value of periodogram
		  	  {
			  	  samples[j].real = periodogram[j];
			  	  samples[j].imag = 0.0;
		  	  }

		  fft(samples,frame_size,twiddle); //do the fft

		  for (j=0 ; j<frame_size ; j++)
			  p_mag[j] = sqrt(samples[j].real*samples[j].real+samples[j].imag*samples[j].imag); //calculate the magnitude
}


void FFT_square()
{
	for (j=0 ; j<frame_size ; j++)
		p_mag_sqr[j]=(p_mag[j]*p_mag[j])/frame_size;  //take the square and then divide by N
}

void power_spectral()
{
	for (j=0 ; j<K; j++)
		PSD[j]=p_mag_sqr[j];  //take the first K value in p_mag_sqr as the PSD
}

void mel_f()
{
	mel_fre[0]=2595.0*log10(1+lower_f/700.0);  //get the first mel frequency
	mel_fre[27]=2595.0*log10(1+upper_f/700.0);  //get the last mel frequency
	avg = (mel_fre[27]-mel_fre[0])/27.0;

	for (j=0 ; j<27; j++)
		mel_fre[j+1]=mel_fre[j]+avg;  //fill in the mel_fre array
}

void real_f()
{
	for (j=0 ; j<28; j++)
		real_fre[j]=700.0*(pow(10,mel_fre[j]/2595.0)-1.0);  //transfer the mel_fre to the original frequency and fill in the array
}

void filterbank()
{
	/*for (j=0; j<filter_num; j++)
		{
			energy[j]=0;
			log_energy[j]=0;
		}*/

	for (m=1; m<27; m++) //filter number
	{
		for (k=1; k<K+1; k++)
		{
			 if (k*avgk<real_fre[m-1])
				 energy[m-1]+=PSD[k-1]*0;

			 if (k*avgk>=real_fre[m-1] && k*avgk<=real_fre[m])
				 energy[m-1]+=PSD[k-1]*(k*avgk-real_fre[m-1])/(real_fre[m]-real_fre[m-1]);

			 if (k*avgk>=real_fre[m] && k*avgk<=real_fre[m+1])
				 energy[m-1]+=PSD[k-1]*(real_fre[m+1]-k*avgk)/(real_fre[m+1]-real_fre[m]);

			 if (k*avgk>real_fre[m+1])
				 energy[m-1]+=PSD[k-1]*0;
		}
	}
}

void filerPSD()
{
	for (j=0; j<filter_num; j++)
		log_energy[j]=log10(energy[j]);  //take the log on the energy array
}

void MFCC_transform()
{
	/*for (j=0; j<13; j++)
		MFCC[j]=0;*/

	for (j=1; j<MFCC_size+1; j++) //iterate from 1 to Z, which is 13
	{
		for (i=1; i<27; i++)
			MFCC[j-1]+=log_energy[i-1]*cos(j*(i-0.5)*PI/26.0); //MFCC formula
	}
}


void syc_initiation()
{
	for (j = 0; j < full_size; j++)
		frame[j] = 0;

	for (j = 0; j< 3; j++)
		scout_squad[j] = 0;

}

int signal_check()
{
	if (current > 2)  //skip the first three data points
	{
		scout_squad[g] = input_left_sample();
		g++;
	}
	if (g == 3)
	{
		if ((scout_squad[2] - scout_squad[0])*(scout_squad[2] - scout_squad[0])>THRESHOLD)
		{
			g = 0;
			for (j = 0; j< 3; j++)
				scout_squad[j] = 0;
			return 1;
		}
		else
		{
			g = 0;
			for (j = 0; j< 3; j++)
				scout_squad[j] = 0;
			return 0;
		}
	}
	return 0;
}

void reboot_inquiry()
{
	char temp_c = '\0';
	printf("Type something to resume to Menu. \n");
	scanf(" %c", &temp_c);
	printf ("***************************************\n");
	if (temp_c != '\0')
	{
		initialization();
		syc_initiation();
		iterator = 0;
		signal = 0;
		check = 0;
		inst = 0;
	}
}

void sychronization()
{
	if (check == 0)
		check = signal_check();
	if (check == 1) // if find the start of the signal
	{
		current = 0;
		if (signal == 1)
			frame[iterator] = input_left_sample();
		iterator++;
		if (iterator == full_size)
			iterator = full_size;
	}
	output_left_sample(0);
	current++;
}

void MFCC_extraction_process() //calculate the MFCC of the entire input signal and store them in 2d MFCC_TOTAL array
{
	for (frame_iterator=0; frame_iterator<frame_num; frame_iterator++)
	{
		for (j=0; j<frame_size; j++)
			frame_seg[j]=frame[frame_iterator*frame_size+j];

		Hamming_window();  //apply the Hamming window
    	Periodogram();  	 //calculate the periodogram
    	FFT();  			 //take FFT over the periodogram
    	FFT_square(); 	 // take square and divide by N of FFT values from FFT()
    	power_spectral();  // take first K values of FFT square to PSD
    	mel_f();
		real_f();
		filterbank();		 //build up the filterbank
		filerPSD();		 //filter through the filterbank
		MFCC_transform();	 //MFCC calculation

		for (j=0; j<13; j++)
			MFCC_TOTAL[frame_iterator][j]=MFCC[j];

		Partial_initialization(); //reinitialize those arrays to calculate MFCC value of next frame_seg
	}
}

void library_initiation()
{
	free_space = data_base_size;
	for (j = 0; j < data_base_size; j++)
	{
		Library[j].vocab = (char *)malloc(5*sizeof(char));
		Library[j].vocab = NULL;
		Library[j].name = (char *)malloc(5*sizeof(char));
		Library[j].name = NULL;
	}

	int lib_ite; //iterator

	for (lib_ite=0; lib_ite<data_base_size; lib_ite++)
		for (j=0; j<frame_num; j++)
			for (n=0; n<MFCC_size; n++)
				Library[lib_ite].MFCC_info[j][n]=0;  //initialize library data
}

void data_base_construction(int num, char* word)
{
	if (free_space == data_base_size)
		library_initiation();

	if (free_space != 0)
	{
		for (j=0; j<frame_num; j++)
		{
			for (n=0; n<MFCC_size; n++)
				Library[num-1].MFCC_info[j][n]=MFCC_TOTAL[j][n];
		}
		Library[num-1].vocab = word;

		printf("Word %s has been successfully constructed!\n",Library[num-1].vocab);
		printf("Please specify the speaker: \n");
		Library[num-1].name = (char *)malloc(8*sizeof(char));
		scanf("%s",&(Library[num-1].name[0]));
		printf("Name: %s\n", Library[num-1].name);
		free_space--;
		printf("Get it! There are %d slot left in the data base. \n", free_space);
	}
	else
	{
		printf("All data base slots have been filled! \n");
		char temp_inst;
		printf("Do you want to clear the library? (Y/N) \n");
		scanf("%c",&temp_inst);
		if (temp_inst == 'Y' || temp_inst == 'y')
			clear_library();
	}
}

void file_name_load() //get all the file names
{
	file_name[0] = (char *)malloc(8*sizeof(char));
	file_name[0] = "C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_1.txt";
	file_name[1] = (char *)malloc(8*sizeof(char));
	file_name[1] = "C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_2.txt";
	file_name[2] = (char *)malloc(8*sizeof(char));
	file_name[2] = "C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_3.txt";
	file_name[3] = (char *)malloc(8*sizeof(char));
	file_name[3] = "C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_4.txt";
	file_name[4] = (char *)malloc(8*sizeof(char));
	file_name[4] = "C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_5.txt";
}

void memory_write(int lib_num)
{
	FILE *read_file;
	read_file = fopen(file_name[lib_num-1], "w");

	if (read_file == NULL)
	{
		printf("Error opening file!\n");
		return;
	}

	fprintf(read_file, "%s\n", Library[lib_num-1].vocab);
	fprintf(read_file, "%s\n", Library[lib_num-1].name);

	for (j=0; j<frame_num; j++)
		{
		for (n=0; n<MFCC_size; n++)
			fprintf(read_file, "%f\n", Library[lib_num-1].MFCC_info[j][n]);
		}
	printf("Memory writing finish!\n");

	fclose(read_file);
}

void memory_load ()
{
	if (free_space == data_base_size)
	{
		printf("library is empty. Memory Loading is allowed! \n");
		printf("Loading process initiated. Processing... \n");
		FILE *load_file;
		file_name_load();
		for (j =0; j< data_base_size; j++)
		{
			load_file = fopen(file_name[j], "r");
			if (load_file == NULL)
				{
					printf("Error opening file!\n");
					return;
				}
			fscanf(load_file, "%c", Library[j].vocab); //load the first line of the file (the vocab info)
			fscanf(load_file, "%c", Library[j].name);  // load the second line of the file (the name)
			int temp = 0;
			while(temp!=MFCC_size*frame_num-1)
			{
				fscanf(load_file, "%f", Library[j].MFCC_info[temp]); //load the MFCC info in the file into the library
				temp ++;// from the third line to the end
			}
			fclose(load_file);
			free_space --;
		}
		printf("Loading process Finish! \n");
	}
	else
	{
		printf("library is not empty! Memory Loading is aborted! \n");
		printf("Please clear out the library first. \n");
	}

}

void clear_library()
{
	library_initiation();
	memoryLoad = 0;
	free_space = data_base_size;
	printf("Library is cleared! \n");
}

void delta_f()
{
    for (j = 0; j<MFCC_size; j++)
	{
		delta = 0;
		for (n = 1 ;n<=2; n++)
		{
			idx1 = j - n;
			idx2 = j + n;
			if (idx1 < 0)
			{
				idx1 = 0;
			}
			if (idx2 > MFCC_size-1)
			{
				idx2 = MFCC_size-1;
			}
			delta += n*(MFCC[idx2]-MFCC[idx1]);
		}
		MFCC[j] += (delta/(10.0*3.0));
		d_array[j] = delta/10.0;
	}
}
void delta_delta()
{
	for (j = 0; j<13; j++)
	{
		delta = 0;
		for (n = 1; n<=2; n++)
		{
			idx1 = j - n;
			idx2 = j + n;
			if (idx1 < 0)
			{
				idx1 = 0;
			}
			if (idx2 > MFCC_size-1)
			{
				idx2 = MFCC_size-1;
			}
			delta += n*(d_array[idx2]-d_array[idx1]);
		}
		MFCC[j] += (delta/(10.0*6.0));
		d_d_array[j] = (delta)/10.0; // note
	}
}

void DTW_process()
{
            for (k = 0; k < data_base_size;k++)
			{
				for (p = 0;p < frame_num;p++)
				{
					for (i = 0; i< MFCC_size; i++) //y- avg
					{
						for (j = 0;j<MFCC_size; j++) //x - new
						{
							DTW[k][p][i][j] = 0;
							dist[k][p][i][j] = ((MFCC_TOTAL[p][j]-Library[k].MFCC_info[p][i]))*((MFCC_TOTAL[p][j]-Library[k].MFCC_info[p][i]));
						}
					}
				}
			}

			// Dynamic Time Warping calculations per test character
			for (k = 0; k <data_base_size;k++ )
			{
				for (p = 0; p < frame_num; p++)
				{
					DTW[k][p][0][0] = dist[k][p][0][0];
					for (i = 1; i<MFCC_size;i++)
					{
						DTW[k][p][i][0] = dist[k][p][i][0] + DTW[k][p][i-1][0];
					}
					for (j = 1; j<MFCC_size;j++)
					{
						DTW[k][p][0][j] = dist[k][p][0][j] + DTW[k][p][0][j-1];
					}
					for (i = 1; i<MFCC_size;i++)
					{
						for (j = 1; j<MFCC_size;j++)
						{
							distance = dist[k][p][i][j];

							DTW_local[1] = DTW[k][p][i][j-1] + distance;
							DTW_local[0] = DTW[k][p][i-1][j-1] + distance;
							DTW_local[2] = DTW[k][p][i-1][j] + distance;

							DTW[k][p][i][j] = min_func(DTW_local[0], DTW_local[1], DTW_local[2]);
						}

					}
				}
			}
//END OF DTW Matrix
}

float min_func(float x, float y, float z) // gets the three values from DTW matrix and returns the least value with var
{
	float var = x;
	if (var > y)
		var = y;
	if (var > z)
		var = z;
	return var; // return the least value
}

void least_cost () // after generating matrices with MFCC for DTW calculation format, and find the least distance by comparing
{
			// Backtracking to find least costly (optimal) path of similiar MFCC values
			for (k = 0; k<data_base_size; k++)
			{
				path_cost[k] = 0;
				for (p = 0; p< frame_num;p++)
				{
					path[k][p][0][0] = MFCC_size-1;
					path[k][p][0][1] = MFCC_size-1;
					pathx = 0;
					y = MFCC_size-1;
					x = MFCC_size-1;
					while (y>0 || x>0)
					{
						pathx++;
						if (y == 0)
							x--;
						else if (x == 0)
							y--;
						else
						{
							if (DTW[k][p][y-1][x] == min_func(DTW[k][p][y-1][x-1], DTW[k][p][y][x-1],DTW[k][p][y-1][x]))
								y--;
							else if (DTW[k][p][y][x-1] == min_func(DTW[k][p][y-1][x-1], DTW[k][p][y][x-1],DTW[k][p][y-1][x]))
								x--;
							else
							{
								y--;
								x--;
							}
						}
						path[k][p][pathx][0] = x; // x <- new - new_mfcc_index
						path[k][p][pathx][1] = y; // y <- avg - avg_mfcc_index
					}
					pathx++;
					path[k][p][pathx][0] = 0; // x <- new
					path[k][p][pathx][1] = 0; // y <- avg
					path_length = pathx+1;
					path_lengths_records[k][p] = path_length;
					for (y = 0; y<path_length;y++)
					{
						path_cost[k] += dist[k][p][(path[k][p][y][1])][(path[k][p][y][0])];
					}
				}				//path_cost[k] /=1000;
			}
			DTW_min = path_cost[0];
			min_index = 0;
			for (k = 0; k<data_base_size; k++)
			{
				if (path_cost[k] < DTW_min)
				{
					DTW_min = path_cost[k];
					min_index = k;
				}
			}
}





