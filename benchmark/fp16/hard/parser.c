#define _CRT_SECURE_NO_WARNINGS 
#include <stdio.h>

int main()
{
	FILE *fp;
	fp = fopen("log","r");
	char tmp;
	int cnt =1;
	if(fp ==NULL){
		exit(0);
	}
	while (fscanf(fp,"%c",&tmp)!=EOF){
		if(tmp =='\n')
			cnt++;
		}
	
	unsigned *ptr;
	ptr = (unsigned*)malloc(sizeof(unsigned)*cnt);
	


	fp = fopen("log","r");
	int i = 0;	
	while(fscanf(fp,"%d",ptr+i)!=EOF){
		i++;
	}
	for(i=0;i<cnt;i++)
		printf("%d\n",ptr[i]);
	

	return 0;
}

