# 📌 Toolchain configuration
CC      = x86_64-elf-gcc
LD      = x86_64-elf-ld
CFLAGS  = -ffreestanding -fno-pie -fno-pic -nostdlib -I.
ASFLAGS = $(CFLAGS)
LDFLAGS = -T kernel.ld -nostdlib -no-pie

# 🔸 object files
OBJS = kernel.o serial.o fb.o font.o font_8x16.o memory.o kprintf.o pmm.o string.o vmm.o \
       idt.o isr.o isr_stubs.o

TARGET  = kernel.elf
SIGNKEY = priv.pem
SIGNOUT = kernel.sig

# 🛠️ Default build: build → sign  (clean은 필요할 때만)
all: $(TARGET) sign

# 🔸 Compile .c files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 🔸 Compile .S (assembly, GAS) files
%.o: %.S
	$(CC) $(ASFLAGS) -c $< -o $@

# 🔸 Link
$(TARGET): $(OBJS) kernel.ld
	$(LD) $(LDFLAGS) $(OBJS) -o $(TARGET)
	@echo "✅ Build complete: $(TARGET)"

# 🔐 Sign ELF binary
sign: $(TARGET)
	openssl dgst -sha256 -sign $(SIGNKEY) -out $(SIGNOUT) $(TARGET)
	@echo "🔐 Signed kernel: $(SIGNOUT)"

# 🧹 Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(SIGNOUT)
	@echo "🧹 Clean done"

.PHONY: all clean sign
