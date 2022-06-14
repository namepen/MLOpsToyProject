# LudWig

이 디렉터리의 소스코드는 오픈소스 AutoML 라이브러리 [LudWig](https://ludwig-ai.github.io/ludwig-docs/)를 사용합니다.

## 환경

`python >= 3.8.9` 의 가상환경 사용을 권장합니다.

### macOS

- [ ] Intel MacOS
- [x] Apple Silicon MacOS

```bash
python3 -m pip install --upgrade pip
```

`LudWig`는 `hdf5`를 사용합니다. homebrew 를 통해 `hdf5`를 설치합니다.
```bash
xcode-select --install
sudo chown -R $(whoami) $(brew --prefix)/*
brew install hdf5
```

`LudWig`는 Rust 컴파일러를 사용합니다. Rust 컴파일러를 설치합니다.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

`LudWig` 를 설치합니다.
```
python3 -m pip install ludwig
```

### Windows

`TODO`

### Linux

`TODO`