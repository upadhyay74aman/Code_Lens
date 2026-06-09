package main

import (
	"reflect"
	"testing"
)

func TestToUTF8Ignore(t *testing.T) {
	input := []byte("hello \xff\xfe world")
	expected := "hello  world"
	result := toUTF8Ignore(input)
	if result != expected {
		t.Errorf("Expected %q, got %q", expected, result)
	}
}

func TestFindLineNumbers(t *testing.T) {
	content := "line1\nline2\nline3\nline4\n"
	chunks := []string{"line2", "line4"}
	expected := []int{2, 4}
	result := findLineNumbers(content, chunks)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestBuildFileTree(t *testing.T) {
	paths := []string{
		"a/b/c.py",
		"a/d.js",
		"e.go",
	}
	tree := buildFileTree(paths)
	if len(tree) != 2 {
		t.Fatalf("Expected 2 root nodes, got %d", len(tree))
	}
	if tree[0].Name != "a" || tree[0].Type != "directory" {
		t.Errorf("Expected first node to be directory 'a', got %s (%s)", tree[0].Name, tree[0].Type)
	}
	if len(tree[0].Children) != 2 {
		t.Errorf("Expected directory 'a' to have 2 children, got %d", len(tree[0].Children))
	}
	if tree[1].Name != "e.go" || tree[1].Type != "file" {
		t.Errorf("Expected second node to be file 'e.go', got %s (%s)", tree[1].Name, tree[1].Type)
	}
}

func TestTextSplitter(t *testing.T) {
	ts := NewTextSplitter(10, 2)
	text := "12345\n\n67890\n\nabcde"
	splits := ts.SplitText(text)
	if len(splits) != 3 {
		t.Errorf("Expected 3 splits, got %d: %v", len(splits), splits)
	}
	if splits[0] != "12345" || splits[1] != "67890" || splits[2] != "abcde" {
		t.Errorf("Unexpected splits: %v", splits)
	}
}
