# Chess Engine

A simple chess engine.

See `python/README.md` for instructions on the Python API.

## Build Instructions

### 1. **Building the C++ Chess Engine**
   To build the C++ chess engine, follow these steps:
   1. Ensure you have the required dependencies installed:
      - A C++17-compatible compiler (e.g., `g++`, `clang`)
      - CMake (version 3.10 or higher)
   2. Run the following commands in the project root:
      ```bash
      mkdir build
      cd build
      cmake ..
      make
      ```
   3. The compiled binaries will be available in the [build](http://_vscodecontentref_/1) directory.

---

### 2. **Setting Up the Python API**
   To use the Python API:
   1. Navigate to the [python](http://_vscodecontentref_/2) directory:
      ```bash
      cd python
      ```
   2. Install the required Python dependencies:
      ```bash
      pip install -r requirements.txt
      ```
   3. Follow the instructions in [README.md](http://_vscodecontentref_/3) for usage.

---

### 3. **Building and Deploying the React Frontend**
   The React frontend provides a user interface for interacting with the chess engine.

   #### **Build the React App**
   1. Navigate to the React frontend directory:
      ```bash
      cd react-chess-frontend
      ```
   2. Install the required dependencies:
      ```bash
      npm install
      ```
   3. Build the production-ready app:
      ```bash
      npm run build
      ```
   4. The optimized static files will be available in the `dist` directory.

   #### **Deploy the React App**
   You can deploy the React app to a live server or hosting platform:
   - **Static File Server**: Copy the contents of the `dist` folder to your web server (e.g., Nginx, Apache).
   - **Cloud Hosting**: Use platforms like [Vercel](https://vercel.com/), [Netlify](https://www.netlify.com/), or [GitHub Pages](https://pages.github.com/).

---

### 4. **Running the Full Application**
   1. Start the C++ chess engine as a backend service.
   2. Serve the React frontend on a live server.
   3. Use the Python API to interact with the chess engine programmatically.

---

For additional details, refer to the documentation in the respective subdirectories.