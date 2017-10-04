
/********************************************************************
 ************* LCDK Number: 15 **********
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include "L138_LCDK_aic3106_init.h"
//#include "L138_LCDK_switch_led.h"
#include "L138_LCDK_aic3106_init.h"
#include "evmomapl138_gpio.h"
#include "fft.h"

/********************************************************************
 ************* Adjustable Parameters **********
 *******************************************************************/
#define frame_size 256 //single frame size (32ms)
#define frame_num 32 //length of input to analyze
#define SAMPLING_f 8000 //remember to change the value in the initiation process
#define THRESHOLD 1000000 //Sensitivity of the microphone
#define data_base_size 31 //size of the library (31)

/********************************************************************
 ************* Constant Parameters **********
 *******************************************************************/
#define filter_num 26
#define MFCC_size 13
#define full_size frame_num*frame_size //valid signal sze (1.024s)
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
int p = 0; //general iterator
int k=0; //General iterator
int m=0; //filter number iterator
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
int g = 0; //signal check iterator iterator

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
int frame_iterator = 0;

/********************************************************************
 ************* Variables and Arrays for DTW **********
 *******************************************************************/

float dist[data_base_size][frame_num][MFCC_size][MFCC_size];
#pragma DATA_SECTION(dist,".EXT_RAM")

float DTW[data_base_size][frame_num][MFCC_size][MFCC_size];
#pragma DATA_SECTION(DTW,".EXT_RAM")

// Resulting  of least costly (distance) between MFCC values
// Contains indices of those match ups
int trace[data_base_size][frame_num][MFCC_size*2][2];
#pragma DATA_SECTION(trace,".EXT_RAM")

// Records the lengths of the traces per window per character
int trace_lengths_records[data_base_size][frame_num];
#pragma DATA_SECTION(trace_lengths_records,".EXT_RAM")

// Records the trace_cost for each character
// used to determine what character is detected by the minimum trace_cost
float trace_cost[data_base_size];

// MISC calculation arrays
float DTW_local[5];

//delta and delta delta arrays
float d_array[MFCC_size];
float d_d_array[MFCC_size];

int traces = 0;
int trace_length = 0;
int x; //x axis of the DTW matrix
int y; //y axis of the DTw matrix

float distance = 0.0; //DTW distance
float DTW_min = 0.0;  //the smallest DTW differences

// delta index
float delta;
int idx1 = 0;
int idx2 = 0;

/********************************************************************
 ************* Variables and Arrays for Data base construction **********
 *******************************************************************/

struct Data {
	char *vocab;
	char *name;
	int correction_times;
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
 ************* Variables for testing and debug **********
 *******************************************************************/

float Total_trial = 0;
float successful_trial = 0;
float successful_speaker = 0;
float successful_vocab = 0;
float successful_r = 0;
float successful_rs = 0;
float successful_rv = 0;

/********************************************************************
 ************* Synchronization Functions **********
 *******************************************************************/

int signal_check(); //detect the start of a signal
void syc_initiation();  //initialize arrays and parameter used in synchronization
void sychronization();  //synchronization interrupt
void sychronization_process(); //synchronization main

/********************************************************************
 ************* Feature extraction Functions **********
 *******************************************************************/

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
void library_display();

/********************************************************************
 ************* Machine Learning Functions **********
 *******************************************************************/

int machine_learning();

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
int least_cost();

void delta_f();
void delta_delta();
void DTW_process();
void DTW_initialization();

/********************************************************************
 ************* Main and Interrupt **********
 *******************************************************************/

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
	if (signal == 0)
	{
		  printf("*********VOICE RECOGNITION MACHINE CEREAL VERSION 1.4**************\n");
		  printf("Command Option:\n");
		  printf("1. Voice recognition \n2. Load memory \n3. Library construction\n4. Clear the Library\n5. Display the library infor\n");
		  printf("There are %d free slot left in the library. \n", free_space);
		  printf("Please type in your command, my majesty.\n");
		  scanf("%d", &inst);

		if (inst == 1 || inst == 3)
		{
			signal = 1; //tell the interrupt to start recording

			if (inst == 1 && free_space == data_base_size) //no Library data but user try to do voice recognition
			{
				printf("Command not allowed! There is nothing to compare with! \n");
			}
			else
			{
				printf("Waiting for input...\n");
				initialization(); 			//initialize the array
				syc_initiation();
				L138_initialise_intr(FS_8000_HZ, ADC_GAIN_24DB, DAC_ATTEN_0DB, LCDK_MIC_INPUT);
			}
		}
	}

	if (inst == 2)
	{
		printf("Load memory? (Y/N)\n");
		char temp_ch = '\0';
		scanf(" %c",&temp_ch);
		printf ("\n");
		if (temp_ch == 'y' || temp_ch == 'Y')
		{
			library_initiation();  //initialize the library to load memory
			memory_load();
			memoryLoad = 1;
		}
		reboot_inquiry();
	}

	if (inst == 1 || inst == 3)
	{
		if (inst == 1 && free_space == data_base_size)
		{
			printf("Extraction process fails!\n");
			reboot_inquiry();
		}
		else
		{
			if (iterator == full_size) //&& memoryLoad == 1) // if it did not detect the start of the input, i wont't increment to 1024
			{
				printf("get it! Processing....\n");
				MFCC_extraction_process();
				printf("process finish. \n");
				if (inst == 1)
				{
					Total_trial++;
					int match_index = 0;
					DTW_initialization();
					DTW_process();
					match_index = least_cost();
					printf("Your word is: %s \n", Library[match_index].vocab);
					printf("The speaker is: %s \n", Library[match_index].name);
					printf("Is the result correct? (Y/N)\n");
					char command1 = '\0';
					scanf( " %c", &command1);
					if (command1 == 'y' || command1 == 'Y')
					{
						successful_trial ++;
						successful_speaker ++;
						successful_vocab ++;

					}
					else if (command1 == 'n' || command1 == 'N')
					{
						printf("Please indicate which one is incorrect: s for speaker, v for vocab and b for both. \n");
						char command2 = '\0';
						scanf(" %c", &command2);
						if (command2 == 'S' || command2 == 's')
							successful_vocab ++;
						else if (command2 == 'v' || command2 == 'V')
							successful_speaker ++;
						else if (command2 == 'b' || command2 == 'B')
							printf("oops \n");
						else
							printf("Invalid command! Try again! \n");

						memory_write(machine_learning());
					}
					else
						printf("Invalid command! Try again! \n");

					successful_r  = (successful_trial)/Total_trial*100.0;
					successful_rs = (successful_speaker)/Total_trial*100.0;
					successful_rv = (successful_vocab)/Total_trial*100.0;
					printf("The total success rate is %f percent \n", successful_r);
					printf("The speaker recognition success rate is %f percent \n", successful_rs);
					printf("The vocabulary recognition success rate is %f percent \n", successful_rv);


				}
				if (inst == 3)
				{
					printf("Library construction starts...\n");
					int data_index = 0;  //specify which vocab to build
					char* temp_word;
					temp_word = (char *)malloc(8*sizeof(char));
					printf("Which Library slot you want to save? (1 to %d) \n", data_base_size);
					printf("(type '0' to discard this input)\n");
					scanf("%d",&data_index);
					if (data_index != 0)
					{
						printf("Which word you just said? \n");
						scanf("%s",&temp_word[0]); // may need a space
						file_name_load(); //load in all file names
						data_base_construction(data_index, temp_word);  //standard vowels data_base construction
					}
					else
						printf("Voice data discarded!\n");
				}
				reboot_inquiry();
			}
		}
	 }
	if (inst == 4)
	{
		clear_library();
		reboot_inquiry();
	}
	if (inst == 5)
	{
		library_display();
		reboot_inquiry();
	}
  }
}


/********************************************************************
 ************* Side functions **********
 *******************************************************************/

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
	//char temp_c;
	/*printf("Type something to resume to Menu. \n");
	scanf(" %c", &temp_c);*/
	printf("Return to the menu.\n");
	printf ("***************** CEREAL TERMINATED **********************\n\n");

	initialization();
	syc_initiation();
	iterator = 0;
	signal = 0;
	check = 0;
	inst = 0;
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
		if (iterator >= full_size)
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
		delta_f();
		delta_delta();
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
		Library[j].vocab = (char *)malloc(10*sizeof(char));
		Library[j].name = (char *)malloc(10*sizeof(char));
	}

	int lib_ite; //iterator

	for (lib_ite=0; lib_ite<data_base_size; lib_ite++)
		for (j=0; j<frame_num; j++)
			for (n=0; n<MFCC_size; n++)
				Library[lib_ite].MFCC_info[j][n]=0;  //initialize library data

	/*for (lib_ite=0; lib_ite<data_base_size; lib_ite++)
		Library[lib_ite].correction_times = 0;*/
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
		memory_write(num);
	}
	else
	{
		printf("All data base slots have been filled! \n");
		printf("Memory Writing fails! \n");
	}
}

void file_name_load() //get all the file names
{
	for (j = 0; j<data_base_size; j++)
	{
		file_name[j] = (char *)malloc(100*sizeof(char));
		char s[1];
		int temp_num = j+1;
		sprintf(s, "%1d", temp_num);
		strcpy(file_name[j], "C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_");
		strcat(file_name[j], s);
		strcat(file_name[j], ".txt");
	}
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
	fprintf(read_file, "%d\n", Library[lib_num-1].correction_times);

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
			fscanf(load_file, "%s", &Library[j].vocab[0]); //load the first line of the file (the vocab info)
			fscanf(load_file, "%s", &Library[j].name[0]);  // load the second line of the file (the name)
			fscanf(load_file, "%d", &Library[j].correction_times);  //// load the third line of the file (the correction_times)
			for (n=0; n< frame_num; n++)
			{
				for (m = 0; m < MFCC_size; m++)
					fscanf(load_file, "%f", &Library[j].MFCC_info[n][m]); //load the MFCC info in the file into the library
																		// from the third line to the end
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

void library_display()
{
	if (free_space != data_base_size)
	{
		printf("Library slot #       Vocabulary        Speaker\n");
			for ( j = 0; j< data_base_size; j++)
			{
				printf("      %d                %s            %s", j+1, Library[j].vocab, Library[j].name);
				printf("\n");
			}
	}
	else
	{
		printf("No data to display! The library is Empty!\n");
	}

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

void DTW_initialization()
{
	 for (k = 0; k < data_base_size;k++)
		{
		 trace_cost [k] = 0;
		 for (p = 0;p < frame_num;p++)
			{
				trace_lengths_records [k][p] = 0;

				for (i = 0; i< 2*MFCC_size; i++) //y- avg
				{
					trace[k][p][i][0] = 0;
					trace[k][p][i][1] = 0;
				}

				for (i = 0; i< MFCC_size; i++) //y- avg
				{
					for (j = 0;j<MFCC_size; j++) //x - new
					{
						DTW[k][p][i][j] = 0;
						dist[k][p][i][j] = 0;
					}
				}
			}
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


							DTW_local[0] = DTW[k][p][i-1][j-1] + distance;
							DTW_local[1] = DTW[k][p][i][j-1] + distance;
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

int least_cost () // after generating matrices with MFCC for DTW calculation format, and find the least distance by comparing
{
			// Backtracking to find least costly (optimal) trace of similiar MFCC values
			for (k = 0; k<data_base_size; k++)
			{
				trace_cost[k] = 0;
				for (p = 0; p< frame_num;p++)
				{
					trace[k][p][0][0] = MFCC_size-1;
					trace[k][p][0][1] = MFCC_size-1;
					traces = 0;
					y = MFCC_size-1;
					x = MFCC_size-1;
					while (y>0 || x>0)
					{
						traces++;
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
						trace[k][p][traces][0] = x; // x <- new - new_mfcc_index
						trace[k][p][traces][1] = y; // y <- avg - avg_mfcc_index
					}
					traces++;
					trace[k][p][traces][0] = 0; // x <- new  //may not need this if we initialize the matrix
					trace[k][p][traces][1] = 0; // y <- avg
					trace_length = traces+1;
					trace_lengths_records[k][p] = trace_length;
					for (j = 0; j<trace_length; j++)
					{
						trace_cost[k] += dist[k][p][(trace[k][p][j][1])][(trace[k][p][j][0])]; //initializtize this array
					}
				}
			}

			int min_index = 0;
			DTW_min = trace_cost[0];
			for (k = 0; k<data_base_size; k++)
			{
				if (trace_cost[k] < DTW_min)
				{
					DTW_min = trace_cost[k];
					min_index = k;
				}
			}
			return min_index;
}


int machine_learning()  //return the Library item index we want to modify
{
	int voc_index = 0;
	printf("Please indicate the correct vocabulary index.");
	scanf("%d", &voc_index);
	while (voc_index > data_base_size)
	{
		printf("The correction is not recognizable! The index exceeds the library size! %s\n");
		printf("Please indicate the correct vocabulary index.");
		scanf("%d", &voc_index);
	}

	char temp_char = 'a';
	while (1)
	{
		printf("You are going to correct the word: %s, the speaker %s\n", Library[voc_index-1].vocab,Library[voc_index-1].name);
		printf("Is that the subject you want to correct? (Y/N) \n");
		scanf(" %c", &temp_char);
		if (temp_char == 'y' || temp_char == 'Y')
			break;
		if (temp_char == 'n' || temp_char == 'N')
		{
			printf("Please indicate the correct vocabulary index.");
			scanf("%d", &voc_index);
		}
	}

	Library[voc_index-1].correction_times += 1; //increment the correction times
	int L =	Library[voc_index-1].correction_times;	//keep track how many times we have corrected this subject
	for (j = 0; j<frame_num; j++)
	{
		for (k=0; k< MFCC_size; k++)
		{
			Library[voc_index-1].MFCC_info[j][k] = (Library[voc_index-1].MFCC_info[j][k]*L +MFCC_TOTAL[j][k])/(L+1);
		}
	}
	printf("The new phase is successfully updated! \n");
	return voc_index;
}



