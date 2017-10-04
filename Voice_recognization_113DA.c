
#include <stdio.h>
#include <math.h>
//#include "L138_LCDK_aic3106_init.h"
//#include "L138_LCDK_switch_led.h"
#include "L138_LCDK_aic3106_init.h"
#include "evmomapl138_gpio.h"
#include "fft.h"

#define PI 3.14159265358979
#define SAMPLING_FREQ 8000.0
#define frame_size 256 //single frame size (32ms)
#define frame_num 16
#define full_size frame_num*frame_size //valid signal size (0.512s*2)
#define a 0.54
#define b 0.46
#define K full_size/2+1
#define upper_f 8000.0
#define lower_f 250.0
#define avgk 8000.0/513.0 //delta f in k domain
#define SAMPLING_f 8000
#define THRESHOLD 1000000
#define filter_num 26
#define MFCC_size 13
#define data_base_size 5

int n = 0; //general iterator
int j = 0; //general iterator
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

int m=0; //filter number iterator
int k=0; //filter bank iterator
int vowel_index = 0; //indicate where the minimum Eucliean index locates

/********************************************************************
 ************* Variables and Arrays for Data base construction **********
 *******************************************************************/

struct Data {
	char *vocab;
	char *name;
	float MFCC_info[frame_num][MFCC_size];
};

struct Data Library[data_base_size];  //the library we are going to build contains
									  //data_base size number of vocab
FILE *f;

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
void Euclidean_distance(int sw);

/********************************************************************
 ************* Library Construction Functions **********
 *******************************************************************/

int data_index;  //specify which vocab to build
int free_space = data_base_size;  //used to choose to continue or exit library construction

void library_initiation();
void data_base_construction(int num);
void memory_write(int num);

/********************************************************************
 ************* Other Functions **********
 *******************************************************************/

void initialization();
void reboot_inquiry();
//void machine_learning(int sw);


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
		printf("type in command:\n");
		printf("1. Voice recognition \n3. Library construction\n4. Clear the Library\n");
		printf("There are %d free slot left in the library. \n", free_space);
		scanf("%d", &inst);

		if (inst == 1 || inst == 3)
		{
			signal = 1; //tell the interrupt to start recording

			if (inst == 3)
				printf("Library construction starts...\n");

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
	if (inst == 1 || inst == 3)
	{
		if (inst == 1 && free_space == data_base_size)
		{
			printf("Extraction process fails!\n");
			reboot_inquiry();
		}
		else
		{
			if (iterator == full_size) // if it did not detect the start of the input, i wont't increment to 1024
			{
				printf("get it! Processing...\n");
				MFCC_extraction_process();
				printf("process finish. \n");
				if (inst == 3)
				{
					printf("Which word you want to save? (1,2,3,4) \n");
					scanf("%d",&data_index);
					data_base_construction(data_index);  //standard vowels data_base construction
					printf("which memory file u want to save it for? (1)\n");
					int number = 0;
					scanf("%d",&number);
					memory_write(number);
				}
				reboot_inquiry();
			}
		}
	 }
	else if (inst == 4)
	{
		library_initiation();
		printf("Everything is cleared! \n");
		reboot_inquiry();
	}
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
	printf("Reboot? (2)\n");
	scanf("%d", &inst);
	if (inst == 2)
	{
		initialization();
		syc_initiation();
		iterator = 0;
		signal = 0;
		check = 0;
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
	Library[0].vocab = (char *)malloc(4*sizeof(char));
	Library[0].vocab = "ice";
	Library[1].vocab = (char *)malloc(4*sizeof(char));
	Library[1].vocab = "cat";
	Library[2].vocab = (char *)malloc(4*sizeof(char));
	Library[2].vocab = "car";
	Library[3].vocab = (char *)malloc(5*sizeof(char));
	Library[3].vocab = "kiss";
	Library[4].vocab = (char *)malloc(5*sizeof(char));
	Library[4].vocab = "late";

	int lib_ite; //iterator

	for (lib_ite=0; lib_ite<data_base_size; lib_ite++)
		for (j=0; j<frame_num; j++)
			for (n=0; n<MFCC_size; n++)
				Library[lib_ite].MFCC_info[j][n]=0;  //initialize library data
}

void data_base_construction(int num)
{
	if (free_space == data_base_size)
		library_initiation();

	if (free_space != 0)
	{
		for (j=0; j<frame_num; j++)
		{
			for (n=0; n<MFCC_size; n++)
				Library[num].MFCC_info[j][n]=MFCC_TOTAL[j][n];
		}

		printf("Word %s has been successfully constructed!\n",Library[num].vocab);
		printf("Please specify the speaker: \n");
		Library[num].name = (char *)malloc(8*sizeof(char));
		scanf("%s",&(Library[num].name[0]));
		printf("Name: %s\n", Library[num].name);
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
			library_initiation(); //clear the library data
	}
}

void memory_write(int num)
{
	f = fopen("C:\\Users\\EE113D\\MyProjects\\voice_recongization_project\\memory\\memory_1.txt", "w");

	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "%s\n", Library[num].vocab);
	fprintf(f, "%s\n", Library[num].name);

	int x;
	int y;

	for (x=0; x<frame_num; x++)
		{
		for (y=0; y<MFCC_size; y++)
			fprintf(f, "%f\n", Library[num].MFCC_info[x][y]);
		}
	printf("Memory writting finish!\n");

	fclose(f);
}
