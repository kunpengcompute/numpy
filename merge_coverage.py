#!/usr/bin/env python3
"""
Merge coverage data from multiple compilation targets for dispatch files.

NumPy's CPU dispatch system compiles .dispatch.cpp files multiple times
for different SIMD targets (baseline, SVE, ASIMD, etc.). Each compilation
produces separate .gcda files, but gcovr only reports coverage from one
variant. This script:

1. Runs the standard gcovr to get base coverage
2. For dispatch files with multiple variants, merges their coverage
3. Updates only those files in the base XML (preserving branch coverage)
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


def run_gcovr(build_dir, source_root, output_xml, config_file=None):
    """Run standard gcovr."""
    cmd = ['gcovr']
    if config_file and Path(config_file).exists():
        cmd += ['--config', str(config_file)]
    cmd += [
        '--xml', str(output_xml),
        '--root', str(source_root),
        str(build_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: gcovr failed: {result.stderr}", file=sys.stderr)
        return False
    return True


def run_gcovr_for_target(target_dir, source_root, output_xml):
    """Run gcovr for a specific target directory."""
    cmd = [
        'gcovr',
        '--xml', str(output_xml),
        '--root', str(source_root),
        '--exclude', 'subprojects/.*',
        '--exclude', 'build/.*',
        '--exclude', 'numpy/_core/src/highway/.*',
        '--exclude', 'numpy/_core/src/npysort/.*',
        '--gcov-ignore-parse-errors', 'all',
        str(target_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def find_dispatch_variants(build_dir):
    """Find source files that have multiple compilation variants."""
    build_path = Path(build_dir)
    source_to_dirs = defaultdict(list)
    
    for gcda_file in build_path.rglob('*.gcda'):
        source_name = gcda_file.stem
        if '.dispatch.' in source_name or '_hwy' in source_name:
            source_to_dirs[source_name].append(gcda_file.parent)
    
    return {name: dirs for name, dirs in source_to_dirs.items() if len(dirs) > 1}


def parse_coverage_xml(xml_path):
    """Parse a Cobertura XML file and extract coverage data."""
    if not Path(xml_path).exists():
        return {}
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    coverage_data = {}
    for cls in root.findall('.//class'):
        filename = cls.get('filename')
        if not filename:
            continue
        
        lines = {}
        for line in cls.findall('.//line'):
            line_num = int(line.get('number'))
            hits = int(line.get('hits', 0))
            lines[line_num] = hits
        
        coverage_data[filename] = lines
    
    return coverage_data


def merge_dispatch_coverage(build_dir, source_root, dispatch_variants):
    """Merge coverage for dispatch files across all variants."""
    merged = {}
    
    for source_name, target_dirs in dispatch_variants.items():
        all_coverage = []
        
        for target_dir in target_dirs:
            temp_xml = Path(build_dir) / f'_temp_{target_dir.name}.xml'
            if run_gcovr_for_target(target_dir, source_root, temp_xml):
                coverage = parse_coverage_xml(temp_xml)
                all_coverage.append(coverage)
            if temp_xml.exists():
                temp_xml.unlink()
        
        if not all_coverage:
            continue
        
        # Merge: take max hits for each line across all variants
        file_lines = defaultdict(lambda: defaultdict(int))
        for coverage in all_coverage:
            for filename, lines in coverage.items():
                for line_num, hits in lines.items():
                    file_lines[filename][line_num] = max(
                        file_lines[filename][line_num], hits
                    )
        
        for filename, lines in file_lines.items():
            merged[filename] = dict(lines)
    
    return merged


def update_xml_with_merged_data(xml_path, merged_data):
    """Update the base coverage XML with merged dispatch file data."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    updated_files = []
    
    for cls in root.findall('.//class'):
        filename = cls.get('filename')
        if filename not in merged_data:
            continue
        
        # Update line hits
        for line in cls.findall('.//line'):
            line_num = int(line.get('number'))
            if line_num in merged_data[filename]:
                new_hits = merged_data[filename][line_num]
                old_hits = int(line.get('hits', 0))
                if new_hits > old_hits:
                    line.set('hits', str(new_hits))
        
        updated_files.append(filename)
        
        # Recalculate class-level stats
        total = 0
        covered = 0
        for line in cls.findall('.//line'):
            total += 1
            if int(line.get('hits', 0)) > 0:
                covered += 1
        
        if total > 0:
            cls.set('line-rate', str(covered / total))
        cls.set('lines-covered', str(covered))
        cls.set('lines-valid', str(total))
    
    # Recalculate package-level stats
    for package in root.findall('.//package'):
        pkg_total = 0
        pkg_covered = 0
        for cls in package.findall('.//class'):
            pkg_total += int(cls.get('lines-valid', 0))
            pkg_covered += int(cls.get('lines-covered', 0))
        if pkg_total > 0:
            package.set('line-rate', str(pkg_covered / pkg_total))
        package.set('lines-covered', str(pkg_covered))
        package.set('lines-valid', str(pkg_total))
    
    # Recalculate root-level stats
    total_lines = 0
    covered_lines = 0
    for package in root.findall('.//package'):
        total_lines += int(package.get('lines-valid', 0))
        covered_lines += int(package.get('lines-covered', 0))
    
    if total_lines > 0:
        root.set('line-rate', str(covered_lines / total_lines))
    root.set('lines-covered', str(covered_lines))
    root.set('lines-valid', str(total_lines))
    
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    return updated_files


def main():
    if len(sys.argv) < 3:
        print("Usage: merge_coverage.py <build_dir> <output_xml>")
        sys.exit(1)
    
    build_dir = Path(sys.argv[1])
    output_xml = Path(sys.argv[2])
    source_root = build_dir.parent
    config_file = source_root / 'gcovr.cfg'
    
    # Step 1: Run standard gcovr to get base coverage
    print("Running standard gcovr for base coverage...")
    if not run_gcovr(build_dir, source_root, output_xml, config_file):
        print("Error: Standard gcovr failed", file=sys.stderr)
        sys.exit(1)
    
    base_coverage = parse_coverage_xml(output_xml)
    print(f"Base coverage: {len(base_coverage)} files")
    
    # Step 2: Find dispatch files with multiple variants
    dispatch_variants = find_dispatch_variants(build_dir)
    print(f"Found {len(dispatch_variants)} dispatch files with multiple variants")
    
    if not dispatch_variants:
        print("No dispatch variants to merge, using base coverage")
        return
    
    # Step 3: Merge coverage for dispatch files
    print("Merging dispatch file coverage...")
    merged_data = merge_dispatch_coverage(build_dir, source_root, dispatch_variants)
    print(f"Merged coverage for {len(merged_data)} files")
    
    # Step 4: Update the base XML with merged data
    print("Updating coverage XML...")
    updated_files = update_xml_with_merged_data(output_xml, merged_data)
    print(f"Updated {len(updated_files)} files in coverage XML")
    
    # Print summary
    tree = ET.parse(output_xml)
    root = tree.getroot()
    line_rate = float(root.get('line-rate', 0)) * 100
    covered = root.get('lines-covered', 0)
    total = root.get('lines-valid', 0)
    print(f"Final: {covered}/{total} lines covered ({line_rate:.2f}%)")


if __name__ == '__main__':
    main()
