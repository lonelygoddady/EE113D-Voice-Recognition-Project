
/********************************************************************
 ************* LCDK Number: 15 **********
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "L138_LCDK_switch_led.h"
#include "L138_LCDK_aic3106_init.h"
#include "fft.h"

/********************************************************************
 ************* Adjustable Parameters **********
 *******************************************************************/
#define frame_size 256 //single frame size (32ms)
#define frame_num 32 //length of input to analyze
#define SAMPLING_f 8000 //remember to change the value in the initiation process
#define THRESHOLD 8000000 //Sensitivity of the microphone 2000000
#define data_base_size 71 //size of the library (60) - six ppl: George, Ben, Lyn, Briggs, Yuyu, Brooke
#define sentence_length 30 //max number of commands the system store.
#define trial_num 5 //how many trials are given to the user to pass the voice recog test
#define admin_status 0 //grant the user the admin status or not when they enter (0 for not, 1 for yes)

/********************************************************************
 ************* Constant Parameters **********
 *******************************************************************/

#define filter_num 26
#define MFCC_size 13
#define full_size frame_num*frame_size //valid signal size (1.024s)
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

/********************************************************************
 ************* Variables and Arrays for sample collections **********
 *******************************************************************/

int16_t frame[full_size]; //the array that contain the entire valid signal input (1.024s)
#pragma DATA_SECTION(frame, ".EXT_RAM")

int16_t frame_seg[frame_size]; //256 samples, 32ms frame segment
int16_t scout_squad[3]; //window use for STE (short time energy)

int current = 0; //record how many invalid data has passed
int check = 0;
int iterator = 0; //iterator used in interrupt so dont reuse it
int inst = 1; //type-in command  (start at one enter the voice recognition mode directly)
int i=0; //sample
int counter = 0; //time counter
int signal = 0; //signal to start recording
int g = 0; //signal check iterator iterator

/********************************************************************
 ************* Variables and Arrays for feature extraction **********
 *******************************************************************/

COMPLEX twiddle[frame_size];  //array used in FFT calculation
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
float Eucliean_d_sqr[4]; //the square of Ecucliean dist

int vowel_index = 0; //indicate where the minimum Eucliean index locates
int memoryLoad = 0;  //signal whether the library has been loaded or not 
int frame_iterator = 0;  //iterate throygh 32 frames of one speech input 

/********************************************************************
 ************* Variables and Arrays for DTW **********
 *******************************************************************/

// receives the difference between MFCC values of template and input feature vector
float match_result[data_base_size][frame_num][MFCC_size][MFCC_size]; //
#pragma DATA_SECTION(match_result,".EXT_RAM")

// mirrors MFCC values from the match result to create warping path in DTW matrix
float DTW[data_base_size][frame_num][MFCC_size][MFCC_size];
#pragma DATA_SECTION(DTW,".EXT_RAM")

// backtracks lowest difference between MFCC values
// includes information of indicies of minimum deviation
int trace[data_base_size][frame_num][MFCC_size*2][2];
#pragma DATA_SECTION(trace,".EXT_RAM")

// Records level of deviation after the matching from DTW matrix by backtracking the warping path
// used to determine what word has been spoken with minimum index
float trace_cost[data_base_size];

// array to construct local comparison mechanism under conditional statement
float DTW_match[5];

//delta and delta delta arrays
float d_array[MFCC_size];
float d_d_array[MFCC_size];

int traces = 0;
int trace_length = 0;
int x; //x coordinates of the DTW matrix
int y; //y coordinates of the DTw matrix

float gap = 0.0;      //DTW distance
float DTW_min = 0.0;  //the smallest DTW differences received from trace_cost

// delta index
float delta;
int idx1 = 0;
int idx2 = 0;

/********************************************************************
 ************* Variables and Arrays for Data base construction **********
 *******************************************************************/

struct Data {  //memory data stored in library (vocabulary, name, correction times and MFCC values)
	char *vocab;
	char *name;
	int correction_times;
	float MFCC_info[frame_num][MFCC_size];
};

struct Data Library[data_base_size];  //the 70-word Library
#pragma DATA_SECTION(Library,".EXT_RAM")

int free_space = data_base_size;  //used to choose to continue or exit library construction
char *file_name[data_base_size];     //store all file names in this array

/********************************************************************
 ************* Variables and Arrays for Machine Learning **********
 *******************************************************************/

char judgement; //human judgment of the vowel
char correction; //human correction

/********************************************************************
 ************* Variables for successful_rate **********
 *******************************************************************/

float Total_trial = 0; //variables to keep track the voice and vocabulary recognition rate
float successful_trial = 0;
float successful_speaker = 0;
float successful_vocab = 0;
float successful_r = 0;
float successful_rs = 0;
float successful_rv = 0;

int admin = admin_status; //indicate the status of the user (admin or not)
int curr_pos = 0; //indicate the curr pos of the stranger
int remain_trial = trial_num; //the user have 4 chances to try
int machine_L = 0; //machine_learning control variable

/********************************************************************
 ************* Variables for Free mode function **********
 *******************************************************************/

char *sentence[sentence_length];
int current_vocab = 0;
int initial_sig = 0;

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

void MFCC_extraction_process(); //extract MFCC values from the input signal
void Partial_initialization(); //initialize some of arrays used in extraction process
void Hamming_window (); //construct hamming window to each frame
void Periodogram();  //apply hamming window to get the periodogram
void FFT(); //Fast fourier transform manipulation
void FFT_square(); //take the square to the FFT values
void power_spectral(); //get the PSD array
void mel_f(); //change from the real frequency scale to the mel frequency scale
void real_f(); //change it back to real frequency scale
void filterbank(); //construct and apply 26 filter banks
void filerPSD(); //calculate logarithm value of the energy array
void MFCC_transform(); //Perform the DCT transform to get MFCC values

/********************************************************************
 ************* Library Construction Functions **********
 *******************************************************************/

void library_initiation(); //initialize the library data
void data_base_construction(int num, char* word); //construct the data base
void clear_library(); //reinitialize the current library
void memory_write(int lib_num); //write extracted features to txt files
void memory_load(); //load the vocab, name, correction times and MFCC information to the system library
void file_name_load(); //tell the system which file needs to be loaded in
void library_display(); //display the library infor stored in the library right now

/********************************************************************
 ************* Machine Learning Functions **********
 *******************************************************************/

int machine_learning(); //perform the machine learning

/********************************************************************
 ************* Voice control function **********
 *******************************************************************/

void free_mode(); //voice control feature
void phase_match(); //algorithm used to detect valid commands combiination

/********************************************************************
 ************* Other Functions **********
 *******************************************************************/

void initialization(); //initialize all arrays used in voice recognition
void reboot_inquiry();  //reinitialize all variabels and notify the user

/********************************************************************
 ************* DTW Functions **********
 *******************************************************************/
float set_small(float x, float y, float z);
int low_dev();

void delta_f(); //delta coefficient generation
void delta_delta(); //delta delta coefficient generation
void DTW_process(); //perform the DTW matrix calculation
void DTW_initialization(); //initialize DTW arrays

/********************************************************************
 ************* Main and Interrupt **********
 *******************************************************************/

interrupt void interrupt4(void) // interrupt service routine
{
	sychronization(); //interrupt algorithm to collect speech samples
	return;
}

int main(void)

{
  LCDK_LED_init();

while (1)
{
	if (signal == 0) //if the intertupt has not been triggered
	{
		if (inst != 6)
		{
		  printf("*********VOICE RECOGNITION MACHINE CEREAL VERSION 2.0**************\n\n");
		  if (admin == 1) //if admin status is detected
		  {
			  printf("Your status: Admin.\n");

			  if (machine_L == 0)
				  printf("Machine learning is off.\n\n");
			  else
				  printf("Machine learning is on.\n\n");

			  printf("Command Option:\n"); //display the menu
			  printf("1. Voice recognition \n2. Load memory \n3. Library construction\n4. Clear the Library\n5. Display the library infor\n6. Voice Control\n7. Turn on/off Machine Learning\n8. Quit admin menu.\n");
			  printf("There are %d free slot left in the library. \n\n", free_space);
			  printf("Please type in your command, my majesty.\n");
			  scanf("%d", &inst);
		  }
		  if (admin == 0) //if admin status is not detected
		  {
			  printf("Your status: Stranger.\n\n");
			  if (curr_pos == 0 && remain_trial == trial_num)
			  {
				  printf("Type something to enter the voice recognition mode.\n");
				  char* str = NULL;
				  scanf("%s", &str);
			  }
			  else
			  {
				  printf("Voice recognition starts.\n");
			  }
		  }
		}

		if (inst == 1 || inst == 3 || inst == 6)
		{
			if (inst == 6 || (inst == 1 && admin ==0))
			{
				if (free_space == data_base_size) //load the library from txt files to the system
				{
					library_initiation();
					if (memoryLoad == 0)
						memory_load();
				}
			}

			signal = 1; //tell the interrupt to start recording

			if (inst != 3 && free_space == data_base_size) //if library has not been loaded
				printf("Command not allowed! There is nothing to compare with! \n");
			else
			{
				if (inst != 6)
				{
					if (admin == 1) //if admin is detected
						printf("Words you can say: cat, no, all, fit, cute, share, crazy, cough, sky, thank\n");
					else if (admin == 0) //if admin status not detected, enter the user verification stage
					{
						printf("You have %d remaining trial.\n",remain_trial);
						if (curr_pos == 0)
						{
							printf("Please say 'cat'.\n"); //verification stage 1
						}
						if (curr_pos == 1)
						{
							printf("Please say 'no'.\n"); //verification stage 2
						}
						if (curr_pos == 2)
						{
							printf("Please say 'crazy'.\n");  //verification stage 3
						}
					}
					printf("Waiting for input...\n");
				}

				initialization(); 			//initialize the array
				syc_initiation();           //initialize the sychronization process
				L138_initialise_intr(FS_8000_HZ, ADC_GAIN_24DB, DAC_ATTEN_0DB, LCDK_MIC_INPUT);
			}
		}
	}

	if (inst == 2) //instruction 2: Load library data from txt files
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

	if (inst == 1 || inst == 3 || inst == 6)
	{
		if (inst == 1 && free_space == data_base_size) //if the library has not been loaded
		{
			printf("Extraction process fails!\n");
			reboot_inquiry();
		}
		else
		{
			if (iterator == full_size) // if it did not detect the start of the input, i wont't increment to 1024
			{
				if (inst != 6)
				{
					printf("get it! Processing....\n");
					MFCC_extraction_process();
					printf("process finish. \n");
				}
				else
					MFCC_extraction_process();

				if (inst == 1)  //instruction 1: voice recognition
				{
					Total_trial++;
					int match_index = 0; //initiate the library matching mechanism
					DTW_initialization();
					DTW_process();
					match_index = low_dev(); //display the system's judgement
					printf("Your word is: %s \n", Library[match_index].vocab);
					printf("The speaker is: %s \n", Library[match_index].name);
					
					if (admin == 1) //if admin is detected
					{
						printf("Is the result correct? Or skip? (Y/N/S)\n"); //ask for users' judgement
						char command1 = '\0';
						scanf(" %c", &command1);
						if (command1 == 'y' || command1 == 'Y')
						{
							successful_trial++;
							successful_speaker++;
							successful_vocab++;

						}
						else if (command1 == 'n' || command1 == 'N')
						{
							printf("Please indicate which one is incorrect: s for speaker, v for vocab and b for both. \n");
							char command2 = '\0';
 							scanf(" %c", &command2); //if the result is not correct, ask the user what is wrong
							if (command2 == 'S' || command2 == 's')
								successful_vocab++;
							else if (command2 == 'v' || command2 == 'V')
								successful_speaker++;
							else if (command2 == 'b' || command2 == 'B')
								printf("oops \n");
							else
								printf("Invalid command! Try again! \n");

							if (machine_L == 1) //if the machine learning is turned on, then do the machine learning
							{
								int A = machine_learning();
								memory_write(A);
							}
						}
						else if (command1 == 's' || command1 == 'S') //if the user choose to skip the input, then ignore this result
						{
							printf("Bad input skipped!\n");
							Total_trial -= 1;
						}
						else
							printf("Invalid command! Try again! \n");
					}
					else
					{
						if (strcmp(Library[match_index].name, "ben") == 0 || strcmp(Library[match_index].name, "Ben") == 0) //admin is set to be Ben
						{
							if (curr_pos == 0) //user verification stage 1
							{
								if (strcmp(Library[match_index].vocab, "cat") == 0 || strcmp(Library[match_index].vocab, "Cat") == 0)
								{
									successful_trial++;
									curr_pos++;
									remain_trial ++;
								}
								else
									remain_trial --;
							}
							if (curr_pos == 1) //user verification stage 2
							{
								if (strcmp(Library[match_index].vocab, "no") == 0 || strcmp(Library[match_index].vocab, "No") == 0)
								{
									successful_trial++;
									curr_pos++;
									remain_trial ++;
								}
								else
									remain_trial --;
							}
							if (curr_pos == 2) //user verification stage 3
							{
								if (strcmp(Library[match_index].vocab, "crazy") == 0 || strcmp(Library[match_index].vocab, "Crazy") == 0)
								{
									successful_trial++;
									curr_pos++;
									remain_trial ++;
								}
								else
									remain_trial --;
							}
						}
						else
							remain_trial --;

						Total_trial++;
						
						if (remain_trial == -1) //if the user has used up all his trials
						{
							printf("You used up all trial chances! Resume to the menu.\n");
							curr_pos = 0;
							remain_trial = trial_num;
						}

						if (curr_pos == 3) //if the user passed all voice recognition tests
						{
							admin = 1;
							inst = 0;
							
							successful_r = 0;
							successful_rv = 0;
							successful_rs = 0;
							successful_trial = 0;
							successful_vocab = 0;
							successful_speaker = 0;
							Total_trial = 0;
							
							printf("Admin authorized!\n");
						}
						if (curr_pos == 4 && admin == 0)
						{
							Total_trial = 0;
							printf("Admin not verified! Start it over!\n");
						}
					}

					successful_r  = (successful_trial)/Total_trial*100.0; //calculate the recognition percentage rate
					successful_rs = (successful_speaker)/Total_trial*100.0;
					successful_rv = (successful_vocab)/Total_trial*100.0;
					
					if (admin == 1) //display the recognition rate
					{
						printf("The total success rate is %f percent \n", successful_r);
						printf("The speaker recognition success rate is %f percent \n", successful_rs);
						printf("The vocabulary recognition success rate is %f percent \n", successful_rv);
					}
				}
				if (inst == 3) //instruction 3: construct the library
				{
					printf("Library construction starts...\n");
					int data_index = 0;  //specify which vocab to build
					char* temp_word;
					temp_word = (char *)malloc(10*sizeof(char));
					printf("Which Library slot you want to save? (1 to %d) \n", data_base_size);
					printf("(type '0' to discard this input)\n");
					scanf("%d",&data_index);
					if (data_index != 0)
					{
						printf("Which word you just said? \n");
						scanf("%s",&temp_word[0]);
						file_name_load(); //load in all file names
						data_base_construction(data_index, temp_word);  // data_base construction initiate
					}
					else
						printf("Voice data discarded!\n");
				}

				if (inst != 6)
				{
					reboot_inquiry(); //resume the menu after one operation
				}
				if (inst == 6) //instruction 6: voice control
				{
					int match_num = 0;
					DTW_initialization();
					DTW_process();
					match_num = low_dev();

					char *temp_vocab;
					temp_vocab = (char *)malloc(10*sizeof(char));
					printf("%s \n", Library[match_num].vocab);
					temp_vocab = " ";
					temp_vocab = Library[match_num].vocab;

					sentence[current_vocab] = (char*)malloc(sentence_length*sizeof(char)); //fill detect commands into sentence

					if (initial_sig == 0)
					{
						for (i = 0; i <sentence_length; i++)
							sentence[i] = NULL;
							initial_sig = 1;
					}

					sentence[current_vocab] = temp_vocab;
					current_vocab ++;

					if (strcmp(temp_vocab, "Finished") == 0) //'Finished' signal the system to operate
					{
						printf("\nYour input sentence is: \n");

						for (j =0; j < current_vocab; j++)
							printf("%s ",sentence[j]);

						printf("\n");

						phase_match();  //check any possible voice control phase combination in the command sentence
						printf("\n \n");
						printf("Voice Control finished.\n");
						initial_sig = 0;
						current_vocab = 0;
						reboot_inquiry();
					}
					else
					{
						initialization();  //continue to take in the next command unless 'Finished' detected
						syc_initiation();
						iterator = 0;
						check = 0;
						signal = 0;
					}
				}
			}
		}
	}

	if (inst == 4) //instruction 4: clear the library used in the system currently
	{
		clear_library();
		reboot_inquiry();
	}

	if (inst == 5) //display the library infor (speaker, vocab, MFCC values) used in the system currently
	{
		library_display();
		reboot_inquiry();
	}

	if (inst == 7) //instruction 7: turn on/off machine learning
	{
		printf("Turn on or turn off Machine Learning? (1 for on and 0 for off)\n");
		scanf("%d", &machine_L);
		reboot_inquiry();
	}

	if (inst == 8) //Instruction 8: quit admin menu and reboot the whole system
	{
		admin = 0;
		inst = 1;
		remain_trial = trial_num;
		curr_pos = 0;
	}
  }
}


/********************************************************************
 ************* Functions **********
 *******************************************************************/

void initialization() //initialize all arrays used in feature extraction process
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

void Partial_initialization() //initialize some array used in the feature extraction process
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
	for (m=1; m<27; m++) //construct 26 filter banks in the mel-frequency scale
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

void MFCC_transform() //perform DCT on the log_energy array
{

	for (j=1; j<MFCC_size+1; j++) //iterate from 1 to Z, which is 13
	{
		for (i=1; i<27; i++)
			MFCC[j-1]+=log_energy[i-1]*cos(j*(i-0.5)*PI/26.0); //MFCC formula
	}
}


void syc_initiation() //initialize arrays used in synchronization
{
	for (j = 0; j < full_size; j++)
		frame[j] = 0;

	for (j = 0; j< 3; j++)
		scout_squad[j] = 0;
}

int signal_check() //STE algorithm to differentiate noise from speech input
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
	printf("Return to the menu.\n");
	printf ("***************** CEREAL TERMINATED **********************\n\n");

	initialization();
	syc_initiation();
	iterator = 0;
	signal = 0;
	check = 0;

	if (admin == 1)
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
    	mel_f();		//build up the mel_frequency scale
		real_f();
		filterbank();		 //build up the filterbank
		filerPSD();		 //filter through the filterbank
		delta_f();	     //apply delta coefficient extraction algorithm
		delta_delta();   //apply the delta delta coefficient extraction algorithm
		MFCC_transform();	 //MFCC calculation

		for (j=0; j<13; j++)
			MFCC_TOTAL[frame_iterator][j]=MFCC[j]; //the arrays stores all MFCC features of the speech input

		Partial_initialization(); //reinitialize those arrays to calculate MFCC value of next frame_seg
	}
}

void library_initiation()
{
	free_space = data_base_size;
	for (j = 0; j < data_base_size; j++)
	{
		Library[j].vocab = (char *)malloc(20*sizeof(char));
		Library[j].name = (char *)malloc(20*sizeof(char));
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
		Library[num-1].correction_times = 0;
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

void file_name_load() //get all the file names of library files
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

	fprintf(read_file, "%s\n", Library[lib_num-1].vocab); //write the vocab name in the first line of library file
	fprintf(read_file, "%s\n", Library[lib_num-1].name);  //write the speaker's name in the second line of library file
	fprintf(read_file, "%d\n", Library[lib_num-1].correction_times); //write the time we perform machine learning in the third line

	for (j=0; j<frame_num; j++)
		{
		for (n=0; n<MFCC_size; n++)
			fprintf(read_file, "%f\n", Library[lib_num-1].MFCC_info[j][n]); //write MFCC values one value a line from line 4
		}
	printf("Memory writing finish!\n");

	fclose(read_file);
}

void memory_load ()
{
	if (free_space == data_base_size )
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
		printf("Library slot #       Vocabulary        Speaker        Correction times\n");
			for ( j = 0; j< data_base_size - free_space; j++)
			{
					printf("      %d                %s                %s            %d", j+1, Library[j].vocab, Library[j].name, Library[j].correction_times);
					printf("\n");
			}
	}
	else
	{
		printf("No data to display! The library is Empty!\n");
	}

}

// adds first term to increase resolution to WMFCC values
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
		MFCC[j] += (delta*0.333333);
		d_array[j] = delta*0.1;
	}
}

// adds second term to increase resolution to WMFCC values
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
		MFCC[j] += (delta*0.01667);
		d_d_array[j] = (delta)*0.1; // note
	}
}

void DTW_initialization()
{
	 for (k = 0; k < data_base_size;k++)
		{
		 trace_cost [k] = 0;
		 for (p = 0;p < frame_num;p++)
			{
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
						match_result[k][p][i][j] = 0;
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
					for (i = 0; i< MFCC_size; i++)   //y -  average template
					{
						for (j = 0;j<MFCC_size; j++) //x - new featured input
						{
							match_result[k][p][i][j] = ((MFCC_TOTAL[p][j]-Library[k].MFCC_info[p][i]))*((MFCC_TOTAL[p][j]-Library[k].MFCC_info[p][i]));
							// record the difference between WMFCC values processed in
                            // newly featured input word spoken (MFCC_TOTAL[p][j])
                            // and library template reference (Library[k].MFCC_info[p][i])
						}
					}
				}
			}

			// Dynamic Time Warping calculations per word input
			for (k = 0; k <data_base_size;k++ )
			{
				for (p = 0; p < frame_num; p++)
				{
					DTW[k][p][0][0] = match_result[k][p][0][0];
					for (i = 1; i<MFCC_size;i++)
					{
						DTW[k][p][i][0] = DTW[k][p][i-1][0] + match_result[k][p][i][0];
                    // initialize axis setting with accumulated WMFCC for
                    // word library template feature vector before matching begins
                    // helps to retrieve minimum match distance indicies for the lowest deviated model calculation
					}
					for (j = 1; j<MFCC_size;j++)
					{
						DTW[k][p][0][j] = DTW[k][p][0][j-1] + match_result[k][p][0][j];
                    // initialize axis setting with accumulated WMFCC for
                    // newly word spoken input feature vector before matching begins
                    // helps to retrieve minimum match distance indicies for the lowest deviated model calculation
					}
					for (i = 1; i<MFCC_size;i++)
					{
						for (j = 1; j<MFCC_size;j++)
						{
							gap = match_result[k][p][i][j];

                            // local weight matching with DTW
							DTW_match[0] = DTW[k][p][i-1][j-1] + gap; // directs indicies to diagonal
							DTW_match[1] = DTW[k][p][i][j-1] + gap;   // directs indicies to proceed above (to Y axis)
							DTW_match[2] = DTW[k][p][i-1][j] + gap;   // directs indicies to proceed right (to X axis)

                            // takes information of local weight mathcing indicies with WMFCC information
                            // outputs to set the minimum of those three values
							DTW[k][p][i][j] = set_small(DTW_match[0], DTW_match[1], DTW_match[2]);

						}

					}
				}
			}
//END OF DTW Matrix
}

// gets the three values from DTW matrix and returns the least value with var
float set_small(float x, float y, float z)
{
	float match = x;

	if (match > y)
		match = y;
	if (match > z)
		match = z;

	return match; // return the least value
}

// after generating matrices with MFCC for DTW calculation format
// find the minimum indicies by backtracking the lowest deviating warping path
int low_dev()
{
			// Backtrack to locate optimal trace of similiar MFCC values
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
							if (DTW[k][p][y-1][x] == set_small(DTW[k][p][y-1][x-1], DTW[k][p][y][x-1],DTW[k][p][y-1][x]))
								y--; // move was toward the y axis
							else if (DTW[k][p][y][x-1] == set_small(DTW[k][p][y-1][x-1], DTW[k][p][y][x-1],DTW[k][p][y-1][x]))
								x--; // move was toward the x axis
							else
							{   // both library and input matched (ideal)
								y--;
								x--;
							}
						}
						trace[k][p][traces][0] = x; // stores how much deviation was to x
						trace[k][p][traces][1] = y; // stores how much deviation was to y
					}
					traces++;
					trace[k][p][traces][0] = 0;
					trace[k][p][traces][1] = 0;
					trace_length = traces+1;


					for (j = 0; j<trace_length; j++)
					{
						trace_cost[k] += match_result[k][p][(trace[k][p][j][1])][(trace[k][p][j][0])];
                        //initializtize this array the lowest WMFCC value difference
                        //with the optimal path prompted from backtracking in this function
					}
				}
			}
			int min_index = 0;
			DTW_min = trace_cost[0];
			for (k = 0; k<data_base_size; k++)
			{
                // compare the elements inside the trace_cost to locate closest match index
				if (trace_cost[k] < DTW_min)
				{
					DTW_min = trace_cost[k];
					min_index = k;
				}
			}
			return min_index; // returns indicies with minimal index
}


int machine_learning()  //return the Library item index we want to modify
{
	int voc_index = 0;
	printf("Please indicate the correct vocabulary index."); // tell number to update the information
	scanf("%d", &voc_index);
	while (voc_index > data_base_size)
	{
		printf("The correction is not recognizable! The index exceeds the library size! %s\n");
		printf("Please indicate the correct vocabulary index.");
		scanf("%d", &voc_index);
	}

	char temp_char = 'a';
	while (1)
	{   // double check before the update is proceed
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

	Library[voc_index-1].correction_times += 1;     //increment the correction times
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

// Phase_match function includes conditional statements regarding voice control
// conditional statements has to be in order while giving commands
// match_count is updated to verify that the command is not jumped from random order
void phase_match()
{
    int	LED_num = 0;
	int match_count = 0;

//command arguments

	for (j = 0; j< sentence_length; j++) //loop through the sentence 
	{
		if (strcmp(sentence[j],"turn") == 0)
		{
			match_count = 1;
		}
		if (strcmp(sentence[j],"on") == 0 && match_count == 1) // "turn on"
		{
			match_count = 2;
		}
		if (strcmp(sentence[j],"off") == 0 && match_count == 1) // "turn off"
		{
			match_count = 3;
		}

		if (strcmp(sentence[j],"LED") == 0 && match_count == 2 )// "turn on LED"
		{
			match_count = 4;
		}
		if (strcmp(sentence[j],"LED") == 0 && match_count == 3 )// "turn off LED"
		{
			match_count = 5;
		}

		if (strcmp(sentence[j],"Display") == 0 )
		{
			match_count = 6;
		}
		if (strcmp(sentence[j],"Library") == 0 && match_count == 6 )// "Display Library"
		{
			match_count = 7;
		}

//LED control feature and arguments

		if (strcmp(sentence[j],"one") == 0)
			LED_num = 4;
		if (strcmp(sentence[j],"two") == 0)
			LED_num = 5;
		if (strcmp(sentence[j],"three") == 0)
			LED_num = 6;
		if (strcmp(sentence[j],"four") == 0)
			LED_num = 7;
		if (strcmp(sentence[j],"all") == 0)
			LED_num = 10;
		if (strcmp(sentence[j],"Finished") == 0)
			break;
	}
	if (match_count == 4) //if certain command pattern is detected
		{
			if (LED_num != 0)
				{
					if (LED_num == 10)
						{
							printf("\nYour command is: turn on all LED\n");
							LCDK_LED_on(4);
							LCDK_LED_on(5);
							LCDK_LED_on(6);
							LCDK_LED_on(7);
							printf("\n All LED get turned on!\n");
						}
					else
						{
							LCDK_LED_on(LED_num);
							printf("\n LED %d gets turned on!",LED_num-3);
						}

				}
			else
				printf("\nNo LED gets turned on!\n");
		}
	else if (match_count == 5) //if certain command pattern is detected
		{
			if (LED_num != 0)
				{
				if (LED_num == 10)
					{
						printf("\nYour command is: turn off LED all\n");
						LCDK_LED_off(4);
						LCDK_LED_off(5);
						LCDK_LED_off(6);
						LCDK_LED_off(7);
					}
				else
					LCDK_LED_off(LED_num);

				printf("\n LED %d gets turned off!",LED_num);
				}
			else
				printf("\nNo LED gets turned off!\n");
		}
// Display feature
	else if (match_count == 7)
	{
			printf("\nYour command is: display Library\n");
			library_display();
	}
	else
	{
		printf("\nNo valid command detected!\n");
	}
}
