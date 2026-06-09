package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

type ChunkResult struct {
	Content    string `json:"content"`
	FilePath   string `json:"file_path"`
	StartLine  int    `json:"start_line"`
	Language   string `json:"language"`
	ChunkIndex int    `json:"chunk_index"`
}

type TreeNode struct {
	Name     string      `json:"name"`
	Type     string      `json:"type"`
	Path     string      `json:"path"`
	Children []*TreeNode `json:"children,omitempty"`
}

type IndexResponse struct {
	Status        string        `json:"status"`
	FilesIndexed  int           `json:"files_indexed"`
	ChunksCreated int           `json:"chunks_created"`
	Chunks        []ChunkResult `json:"chunks"`
	FileTree      []*TreeNode   `json:"file_tree"`
	ReadmeSnippet string        `json:"readme_snippet"`
}

type ErrorResponse struct {
	Status string `json:"status"`
	Detail string `json:"detail"`
}

var skipDirs = map[string]bool{
	"node_modules": true,
	".git":         true,
	"__pycache__":  true,
	"dist":         true,
	"build":        true,
	".next":        true,
	"venv":         true,
	"env":          true,
}

var allowedExtensions = map[string]string{
	".py":    "python",
	".js":    "javascript",
	".ts":    "typescript",
	".jsx":   "jsx",
	".tsx":   "tsx",
	".java":  "java",
	".cpp":   "cpp",
	".c":     "c",
	".cs":    "csharp",
	".go":    "go",
	".rs":    "rust",
	".rb":    "ruby",
	".php":   "php",
	".swift": "swift",
	".kt":    "kotlin",
	".md":    "markdown",
}

func main() {
	repoURL := flag.String("repo-url", "", "Git repository URL")
	branch := flag.String("branch", "main", "Branch to clone")
	tempDir := flag.String("temp-dir", "", "Target folder to clone the repo into")
	chunkSize := flag.Int("chunk-size", 1000, "Chunk size for splitter")
	chunkOverlap := flag.Int("chunk-overlap", 200, "Chunk overlap for splitter")

	flag.Parse()

	if *repoURL == "" || *tempDir == "" {
		writeError("Missing required arguments: --repo-url and --temp-dir must be provided")
		return
	}

	// 1. Clone repository
	err := cloneRepo(*repoURL, *branch, *tempDir)
	if err != nil {
		writeError(fmt.Sprintf("Cloning repository failed: %v", err))
		return
	}

	// 2. Find files to index
	files, err := findFiles(*tempDir)
	if err != nil {
		writeError(fmt.Sprintf("Failed to read repository files: %v", err))
		return
	}

	if len(files) == 0 {
		writeError("No files matching the supported extensions found in this repository (.py, .js, .ts, etc.)")
		return
	}

	// 3. Process each file: split and calculate metadata
	splitter := NewTextSplitter(*chunkSize, *chunkOverlap)
	var allChunks []ChunkResult
	var relativePaths []string

	for _, fullPath := range files {
		relPath, err := filepath.Rel(*tempDir, fullPath)
		if err != nil {
			continue
		}
		// Standardize path to use forward slash
		relPath = filepath.ToSlash(relPath)
		relativePaths = append(relativePaths, relPath)

		ext := strings.ToLower(filepath.Ext(fullPath))
		lang := allowedExtensions[ext]
		if lang == "" {
			lang = "text"
		}

		contentBytes, err := os.ReadFile(fullPath)
		if err != nil {
			continue
		}

		// Sanitize/ignore invalid UTF-8 bytes
		content := toUTF8Ignore(contentBytes)

		// Split file content
		fileChunks := splitter.SplitText(content)

		// Find starting line numbers
		startLines := findLineNumbers(content, fileChunks)

		for idx, chunkText := range fileChunks {
			allChunks = append(allChunks, ChunkResult{
				Content:    chunkText,
				FilePath:   relPath,
				StartLine:  startLines[idx],
				Language:   lang,
				ChunkIndex: idx,
			})
		}
	}

	// 4. Build file tree
	sort.Strings(relativePaths)
	tree := buildFileTree(relativePaths)

	// 5. Read README snippet
	readmeSnippet := ""
	readmePaths := []string{
		filepath.Join(*tempDir, "README.md"),
		filepath.Join(*tempDir, "readme.md"),
		filepath.Join(*tempDir, "Readme.md"),
	}
	for _, rp := range readmePaths {
		if fileInfo, err := os.Stat(rp); err == nil && !fileInfo.IsDir() {
			if data, err := os.ReadFile(rp); err == nil {
				readmeStr := toUTF8Ignore(data)
				if len(readmeStr) > 3000 {
					readmeSnippet = readmeStr[:3000]
				} else {
					readmeSnippet = readmeStr
				}
				break
			}
		}
	}

	// Output successful JSON response
	response := IndexResponse{
		Status:        "success",
		FilesIndexed:  len(files),
		ChunksCreated: len(allChunks),
		Chunks:        allChunks,
		FileTree:      tree,
		ReadmeSnippet: readmeSnippet,
	}

	outBytes, err := json.Marshal(response)
	if err != nil {
		writeError(fmt.Sprintf("Failed to marshal output JSON: %v", err))
		return
	}

	fmt.Print(string(outBytes))
}

func cloneRepo(repoURL, branch, tempDir string) error {
	if _, err := exec.LookPath("git"); err != nil {
		return fmt.Errorf("git command not found on the system. Please make sure git is installed and in your PATH")
	}

	// Try shallow clone depth 1
	cmd := exec.Command("git", "clone", "--depth", "1", "-b", branch, repoURL, tempDir)
	if err := cmd.Run(); err != nil {
		// Fallback to full clone in case shallow clone fails (e.g. branch is not a branch name but a commit hash)
		os.RemoveAll(tempDir)
		cmdFallback := exec.Command("git", "clone", "-b", branch, repoURL, tempDir)
		if err := cmdFallback.Run(); err != nil {
			return fmt.Errorf("git clone failed: %w", err)
		}
	}
	return nil
}

func findFiles(rootDir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(rootDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if _, allowed := allowedExtensions[ext]; allowed {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func toUTF8Ignore(b []byte) string {
	var sb strings.Builder
	sb.Grow(len(b))
	temp := b
	for len(temp) > 0 {
		r, size := utf8.DecodeRune(temp)
		if r == utf8.RuneError && size == 1 {
			temp = temp[1:]
			continue
		}
		sb.WriteRune(r)
		temp = temp[size:]
	}
	return sb.String()
}

func findLineNumbers(content string, chunks []string) []int {
	lineNumbers := make([]int, len(chunks))
	lastPos := 0
	for idx, chunk := range chunks {
		pos := strings.Index(content[lastPos:], chunk)
		if pos != -1 {
			pos += lastPos
			lineNumbers[idx] = strings.Count(content[:pos], "\n") + 1
			lastPos = pos + len(chunk)
		} else {
			pos = strings.Index(content, chunk)
			if pos != -1 {
				lineNumbers[idx] = strings.Count(content[:pos], "\n") + 1
				lastPos = pos + len(chunk)
			} else {
				lineNumbers[idx] = 1
			}
		}
	}
	return lineNumbers
}

func buildFileTree(filePaths []string) []*TreeNode {
	root := []*TreeNode{}
	for _, path := range filePaths {
		parts := strings.Split(path, "/")
		currentLevel := &root
		currentPath := ""

		for i, part := range parts {
			if currentPath == "" {
				currentPath = part
			} else {
				currentPath = currentPath + "/" + part
			}
			isLast := i == len(parts)-1
			nodeType := "directory"
			if isLast {
				nodeType = "file"
			}

			var existingNode *TreeNode
			for _, node := range *currentLevel {
				if node.Name == part {
					existingNode = node
					break
				}
			}

			if existingNode == nil {
				newNode := &TreeNode{
					Name: part,
					Type: nodeType,
					Path: currentPath,
				}
				if !isLast {
					newNode.Children = []*TreeNode{}
				}
				*currentLevel = append(*currentLevel, newNode)
				existingNode = newNode
			}

			if !isLast {
				currentLevel = &existingNode.Children
			}
		}
	}
	return root
}

func writeError(detail string) {
	errResp := ErrorResponse{
		Status: "error",
		Detail: detail,
	}
	b, _ := json.Marshal(errResp)
	fmt.Print(string(b))
}

type TextSplitter struct {
	ChunkSize    int
	ChunkOverlap int
	Separators   []string
}

func NewTextSplitter(size, overlap int) *TextSplitter {
	return &TextSplitter{
		ChunkSize:    size,
		ChunkOverlap: overlap,
		Separators:   []string{"\n\n", "\n", " ", ""},
	}
}

func (ts *TextSplitter) SplitText(text string) []string {
	return ts.splitText(text, ts.Separators)
}

func (ts *TextSplitter) splitText(text string, separators []string) []string {
	if len(text) == 0 {
		return nil
	}

	if len([]rune(text)) <= ts.ChunkSize {
		return []string{text}
	}

	var separator string
	var nextSeparators []string
	found := false

	for i, sep := range separators {
		if sep == "" {
			separator = sep
			nextSeparators = separators[i+1:]
			found = true
			break
		}
		if strings.Contains(text, sep) {
			separator = sep
			nextSeparators = separators[i+1:]
			found = true
			break
		}
	}

	if !found {
		var chars []string
		for _, r := range text {
			chars = append(chars, string(r))
		}
		return chars
	}

	var rawSplits []string
	if separator == "" {
		for _, r := range text {
			rawSplits = append(rawSplits, string(r))
		}
	} else {
		rawSplits = strings.Split(text, separator)
	}

	var finalSplits []string
	for _, part := range rawSplits {
		if len([]rune(part)) <= ts.ChunkSize {
			finalSplits = append(finalSplits, part)
		} else {
			subSplits := ts.splitText(part, nextSeparators)
			finalSplits = append(finalSplits, subSplits...)
		}
	}

	return ts.mergeSplits(finalSplits, separator)
}

func (ts *TextSplitter) mergeSplits(splits []string, separator string) []string {
	var chunks []string
	if len(splits) == 0 {
		return chunks
	}

	calcLength := func(parts []string) int {
		if len(parts) == 0 {
			return 0
		}
		total := 0
		for _, p := range parts {
			total += len([]rune(p))
		}
		total += (len(parts) - 1) * len([]rune(separator))
		return total
	}

	var currentDoc []string
	for _, split := range splits {
		if split == "" {
			continue
		}

		if calcLength(append(currentDoc, split)) > ts.ChunkSize {
			if len(currentDoc) > 0 {
				chunks = append(chunks, strings.Join(currentDoc, separator))

				for len(currentDoc) > 0 && calcLength(append(currentDoc, split)) > ts.ChunkSize {
					currentDoc = currentDoc[1:]
				}
				for len(currentDoc) > 0 && calcLength(currentDoc) > ts.ChunkOverlap {
					currentDoc = currentDoc[1:]
				}
			}
		}
		currentDoc = append(currentDoc, split)
	}

	if len(currentDoc) > 0 {
		chunks = append(chunks, strings.Join(currentDoc, separator))
	}
	return chunks
}
