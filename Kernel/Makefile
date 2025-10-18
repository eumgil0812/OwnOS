TARGET = kernel.elf
OBJS = kernel.o font_8x16.o   # ✅ 폰트 오브젝트 추가

CC = x86_64-elf-gcc
LD = x86_64-elf-ld
CFLAGS = -ffreestanding -m64 -O2 -Wall -Wextra
LDFLAGS = -T kernel.ld --oformat=elf64-x86-64

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

clean:
	rm -f $(OBJS) $(TARGET)
