{
  description = "nix shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, utils, fenix}:
    utils.lib.eachDefaultSystem (system:
    let
        fenixStable = fenix.packages.${system}.stable.withComponents [ "cargo" "clippy" "rust-src" "rustc" "rustfmt" "llvm-tools-preview" ];
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
        };
        in {
          # use clang 11 because nix's clang is 11
          # annoying link errors if we try clang 15
          devShell = pkgs.mkShell.override {} {
            shellHook = ''
              export CARGO_TARGET_DIR="$(git rev-parse --show-toplevel)/target_dirs/nix_rustc";
            '';
            RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
            # DYLD_LIBRARY_PATH = "${pkgs.openblas}/lib";
            # LD_LIBRARY_PATH = "${pkgs.openssl.out}/lib:${pkgs.libtorch-bin}/lib";
            # LIBTORCH="${pkgs.libtorch-bin}";
            buildInputs =
              with pkgs; [
                openblas
                # rust-src
                pkg-config
                fenixStable
                fenix.packages.${system}.rust-analyzer
                just
                cargo-expand
                # cargo
                # rustc
                nix
                nix.dev
                rust-cbindgen # for executable cbindgen
                clang-tools_15 # for up to date clangd
                clang_11
                boost
                protobuf
                pkg-config
                openssl.dev
                openssl
              ] ++
              pkgs.lib.optionals stdenv.isDarwin [
                darwin.apple_sdk.frameworks.Security
                 pkgs.libiconv
                darwin.apple_sdk.frameworks.SystemConfiguration
                darwin.apple_sdk.frameworks.Accelerate
                darwin.apple_sdk.frameworks.QuartzCore
                darwin.apple_sdk.frameworks.MetalPerformanceShaders
              ];
          };
    });
}
