{
  description = "Description for the project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
    eihw-packages.url = "git+https://git.rz.uni-augsburg.de/gerczuma/eihw-packages?ref=main";
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = ["x86_64-linux"];

      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: {
        # Per-system attributes can be defined here. The self' and inputs'
        # module parameters provide easy access to attributes of the same
        # system.

        devenv.shells.default = {
          name = "devenv";

          # https://devenv.sh/reference/options/
          packages = with pkgs; [
            # pkgs.glibc
            gcc-unwrapped.out
            ffmpeg
            alejandra
            git
            lapack.dev
            llvm.dev
            libsndfile.out
            stdenv.cc.cc.lib
            zlib
          ];
          languages = {
            python = {
              version = "3.10";
              enable = true;
              poetry = {
                enable = true;
              };
            };
            nix.enable = true;
          };
          scripts = {
            reinstall-venv.exec = "rm -rf $DEVENV_ROOT/.venv $DEVENV_ROOT/.direnv $DEVENV_ROOT/.devenv && direnv reload";
            SMILExtract.exec = "LD_LIBRARY_PATH= ${inputs'.eihw-packages.legacyPackages.opensmile}/bin/SMILExtract \"$@\"";
            deepspectrum.exec = "LD_LIBRARY_PATH= ${inputs'.eihw-packages.legacyPackages.deepspectrum}/bin/deepspectrum \"$@\"";
          };
        };
      };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.
      };
    };
}
