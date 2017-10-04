#include <stdio.h>
#include <math.h>
#include "L138_LCDK_aic3106_init.h"
#include "evmomapl138_gpio.h"

#define SAMPLING_f 16000
#define frame_size 512 // 32 ms
#define THRESHOLD 100000

int16_t frame[2*frame_size];
#pragma DATA_SECTION(frame, ".EXT_RAM")
int16_t scout_squad [3];

int inst = 0; //type-in command  0:no command 1:start recording 2:reboot
int check = 0;

int i = 0; //iterator used in interrupt so dont reuse it
int j = 0; //general iterator
int k = 0; //general iterator

void initiation ();
int signal_check();

interrupt void interrupt4 (void)
{
	if (check == 0)
		check = signal_check();
	if (check == 1) // if find the start of the signal
	{
		if (inst == 1)
			frame[i] = input_left_sample();
		i++;
		if (i == 1024)
			i = 1024;
	}
	output_left_sample(0);
	return;
}

int main(void)
{
	initiation ();

	while(1)
	{
		if (inst !=1 )
		{
			if ( inst != 2)
			{
				printf("type in command (1)\n");
				scanf("%d", &inst);
			}
			if (inst == 1)
				{
					printf("Waiting for input...\n");
					L138_initialise_intr(FS_16000_HZ,ADC_GAIN_24DB,DAC_ATTEN_0DB,LCDK_MIC_INPUT);
				}
		}
		if (inst == 1)
			{
				if (i == 1024) // if it did not detect the start of the input, i wont't increment to 1024
				{
					printf("get it!\n");
					printf("Reboot? (2)\n");
					scanf("%d", &inst);
					if (inst == 2)
					{
						initiation ();
						i = 0;
						inst = 0;
					}
				}
			}
	}
}


void initiation ()
{
	for (j = 0; j < 2*frame_size; j++)
		frame [j] = 0;

	for (j = 0; j< 3; j++)
		scout_squad [j] = 0;

}

int signal_check()
{
	scout_squad[k] = input_left_sample();
	k++;
	if (k == 3)
	{
		if ((scout_squad[2]-scout_squad[0])^2>THRESHOLD && (scout_squad[2]-scout_squad[1])^2>THRESHOLD)
		{
			k = 0;
			for (j = 0; j< 3; j++)
				scout_squad [j] = 0;
			return 1;
		}
		else
		{
			k = 0;
			for (j = 0; j< 3; j++)
				scout_squad [j] = 0;
			return 0;
		}
	}
	return 0;
}
