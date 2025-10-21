# 📌 Toolchain configuration
CC      = x86_64-elf-gcc
LD      = x86_64-elf-ld
CFLAGS  = -ffreestanding -fno-pie -fno-pic -nostdlib -I.
LDFLAGS = -T kernel.ld -nostdlib -no-pie

# 🔸 object files
OBJS = kernel.o serial.o fb.o font.o font_8x16.o memory.o kprintf.o pmm.o string.o vmm.o


TARGET  = kernel.elf
SIGNKEY = priv.pem
SIGNOUT = kernel.sig

# 🛠️ Default build: clean → build → sign
all: clean $(TARGET) sign

# 🔸 Compile .c files into .o object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 🔸 Link all object files into the final ELF binary
$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(TARGET)
	@echo "✅ Build complete: $(TARGET)"

# 🔐 Sign ELF binary with private key
sign:
	openssl dgst -sha256 -sign $(SIGNKEY) -out $(SIGNOUT) $(TARGET)
	@echo "🔐 Signed kernel: $(SIGNOUT)"

# 🧹 Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(SIGNOUT)
	@echo "🧹 Clean done"

.PHONY: all clean sign
