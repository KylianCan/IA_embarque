RUN docker build -t student .
RUN cp -r inference/ Work/
RUN ./student.sh
RUN cd Work/inference/
RUN python3 main.py
RUN gcc -W -o inference main.c Bmp2Matrix.c Bmp2Matrix.h -lm

