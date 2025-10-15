TARGET = kernel.elf                     # 최종 결과물 파일 이름
OBJS = kernel.o framebuffer.o font8x16.o kprintf.o memory.o   # 중간 산출물 (목표 .o 파일들)

CC = x86_64-elf-gcc                     # 컴파일러
LD = x86_64-elf-ld                      # 링커
CFLAGS = -ffreestanding -m64 -O2 -Wall -Wextra   # 컴파일 옵션
LDFLAGS = -T kernel.ld --oformat=elf64-x86-64    # 링커 옵션 (링커 스크립트 지정)

all: $(TARGET)                          # 기본 명령어 → make 입력 시 실행

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@          # .c 파일을 .o로 컴파일

$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS)      # .o 파일들을 링크해서 kernel.elf 생성

clean:
	rm -f $(OBJS) $(TARGET)             # make clean 하면 중간파일 지움
