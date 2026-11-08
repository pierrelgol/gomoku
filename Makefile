NAME     := Gomoku
ZIG      ?= zig
FORKS    := --fork=vendor/libvaxis --fork=vendor/uucode
SRCS     := $(shell find src assets vendor -type f) build.zig build.zig.zon
OUT      := zig-out/bin/$(NAME)

all: $(NAME)

# Only rebuilt when a source changes (no relink otherwise).
$(NAME): $(SRCS)
	$(ZIG) build $(FORKS) -Doptimize=fast
	cp $(OUT) $(NAME)

test:
	$(ZIG) build $(FORKS) test

bench: $(NAME)
	./$(NAME) --bench

clean:
	rm -rf .zig-cache

fclean: clean
	rm -rf zig-out zig-pkg $(NAME)

re: fclean all

.PHONY: all clean fclean re test bench
