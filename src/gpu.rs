//! Host GPU memory for `GET /v1/aivyx/residency`. No LLM backend reports
//! free or total VRAM (Ollama's `/api/ps` lists only what its own models
//! hold; `llama-server` reports nothing), so the broker reads the GPU
//! itself: `nvidia-smi` (summed across GPUs), else the AMD card with the
//! most VRAM via sysfs. Best-effort: `None` when neither answers.

use std::path::{Path, PathBuf};

pub(crate) const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Vram {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

/// Where to read VRAM from (`--vram-source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum VramSource {
    /// `nvidia-smi`, else AMD sysfs.
    Auto,
    Nvidia,
    Amd,
    /// Report no VRAM.
    None,
}

/// Reads the host's VRAM. A trait so tests can fake it.
pub trait GpuProbe: Send + Sync {
    fn vram(&self) -> Option<Vram>;
}

pub struct SystemGpuProbe {
    source: VramSource,
    drm_root: PathBuf,
}

impl SystemGpuProbe {
    pub fn new(source: VramSource) -> Self {
        SystemGpuProbe {
            source,
            drm_root: PathBuf::from("/sys/class/drm"),
        }
    }
}

impl GpuProbe for SystemGpuProbe {
    fn vram(&self) -> Option<Vram> {
        match self.source {
            VramSource::None => None,
            VramSource::Nvidia => nvidia(),
            VramSource::Amd => amd(&self.drm_root),
            VramSource::Auto => nvidia().or_else(|| amd(&self.drm_root)),
        }
    }
}

fn nvidia() -> Option<Vram> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
}

/// One `"<total MiB>, <used MiB>"` line per GPU, summed.
pub(crate) fn parse_nvidia_smi(out: &str) -> Option<Vram> {
    let mut vram: Option<Vram> = None;
    for line in out.lines().filter(|l| !l.trim().is_empty()) {
        let (total, used) = line.split_once(',')?;
        let total = total.trim().parse::<u64>().ok()? * MIB;
        let used = used.trim().parse::<u64>().ok()? * MIB;
        let sum = vram.get_or_insert(Vram {
            total_bytes: 0,
            used_bytes: 0,
        });
        sum.total_bytes += total;
        sum.used_bytes += used;
    }
    vram
}

/// The `cardN` under `drm_root` with the largest `mem_info_vram_total` (a
/// discrete GPU beside an integrated one). Connector entries (`cardN-DP-1`)
/// and cards without the files are skipped.
pub(crate) fn amd(drm_root: &Path) -> Option<Vram> {
    std::fs::read_dir(drm_root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("card") && !name.contains('-')
        })
        .filter_map(|e| {
            let device = e.path().join("device");
            Some(Vram {
                total_bytes: read_u64(&device.join("mem_info_vram_total"))?,
                used_bytes: read_u64(&device.join("mem_info_vram_used"))?,
            })
        })
        .max_by_key(|v| v.total_bytes)
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_output_sums_every_gpu() {
        assert_eq!(
            parse_nvidia_smi("23028, 689\n"),
            Some(Vram {
                total_bytes: 23028 * MIB,
                used_bytes: 689 * MIB
            })
        );
        assert_eq!(
            parse_nvidia_smi("24564, 1000\n24564, 2000\n\n"),
            Some(Vram {
                total_bytes: 2 * 24564 * MIB,
                used_bytes: 3000 * MIB
            })
        );
        assert_eq!(parse_nvidia_smi(""), None);
        assert_eq!(parse_nvidia_smi("[N/A], [N/A]\n"), None);
    }

    fn card(root: &std::path::Path, name: &str, total: u64, used: u64) {
        let dev = root.join(name).join("device");
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
        std::fs::write(dev.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
    }

    #[test]
    fn amd_sysfs_picks_the_card_with_the_most_vram() {
        let root = tempfile::tempdir().unwrap();
        // An iGPU beside a discrete card, plus a connector entry to ignore.
        card(root.path(), "card0", 2 << 30, 1 << 30);
        card(root.path(), "card1", 16 << 30, 4 << 30);
        std::fs::create_dir_all(root.path().join("card1-DP-1")).unwrap();
        assert_eq!(
            amd(root.path()),
            Some(Vram {
                total_bytes: 16 << 30,
                used_bytes: 4 << 30
            })
        );
    }

    #[test]
    fn amd_sysfs_without_vram_files_is_none() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("card0").join("device")).unwrap();
        assert_eq!(amd(root.path()), None);
        assert_eq!(amd(&root.path().join("missing")), None);
    }

    #[test]
    fn the_none_source_never_probes() {
        assert_eq!(SystemGpuProbe::new(VramSource::None).vram(), None);
    }
}
