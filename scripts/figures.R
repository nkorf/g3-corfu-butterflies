#!/usr/bin/env Rscript
# Publication-quality figures for the G3 Corfu Butterflies paper.
#
# Replaces the matplotlib figures (cardinality.png, long_tail.png,
# cooccurrence.png) with ggplot2 versions rendered at 600 dpi using
# vector-friendly settings. Reads `data/metadata.csv` and
# `data/label_vocab.json` directly so the R pipeline is independent
# of the Python one.
#
# Usage:
#   Rscript scripts/figures.R [data_dir] [out_dir]
#
# Author:  Nikolaos Korfiatis, Ionian University. nkorf@ionio.gr
# License: MIT.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(jsonlite)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "data"
out_dir  <- if (length(args) >= 2) args[2] else "paper/figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

PALETTE_PRIMARY   <- "#355070"
PALETTE_SECONDARY <- "#b56576"
PALETTE_ACCENT    <- "#6d597a"
PALETTE_GRID      <- "grey88"

# ---- shared theme ----------------------------------------------------------
theme_g3 <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "Helvetica") +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(colour = PALETTE_GRID, linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      axis.line          = element_line(colour = "grey20", linewidth = 0.3),
      axis.ticks         = element_line(colour = "grey20", linewidth = 0.3),
      axis.title         = element_text(colour = "grey15"),
      axis.text          = element_text(colour = "grey25"),
      plot.title         = element_text(face = "plain", size = base_size + 1,
                                        margin = margin(b = 4)),
      plot.subtitle      = element_text(colour = "grey35", size = base_size - 1,
                                        margin = margin(b = 6)),
      plot.margin        = margin(6, 8, 6, 6)
    )
}

# ---- load ------------------------------------------------------------------
meta  <- read_csv(file.path(data_dir, "metadata.csv"), show_col_types = FALSE)
vocab <- fromJSON(file.path(data_dir, "label_vocab.json"))$labels

Y <- do.call(
  rbind,
  lapply(strsplit(meta$labels_multihot, " "), function(s) as.integer(s))
)
stopifnot(ncol(Y) == length(vocab))

namespace_of <- function(x) sub(":.*$", "", x)
short_label  <- function(x) sub("^[^:]+:", "", x)

# ---- 1. cardinality distribution ------------------------------------------
# FDLF: discrete count distribution -> bar chart is appropriate. Remove the
# redundant y-axis (the on-bar labels already carry the count) and emphasise
# the modal cardinality with a darker fill so the eye lands on the story.
card <- rowSums(Y)
card_df <- as.data.frame(table(card)) |>
  mutate(card = as.integer(as.character(card)),
         Freq = as.integer(Freq),
         is_mode = Freq == max(Freq))

mean_card <- mean(card)
max_card  <- max(card)
n_imgs    <- sum(card_df$Freq)

p_card <- ggplot(card_df, aes(x = card, y = Freq)) +
  geom_col(aes(fill = is_mode), width = 0.78) +
  geom_text(aes(label = Freq), vjust = -0.6, size = 2.7, colour = "grey15") +
  geom_vline(xintercept = mean_card, linetype = "22",
             colour = "grey35", linewidth = 0.35) +
  annotate("text", x = mean_card, y = max(card_df$Freq) * 1.05,
           label = sprintf("mean = %.2f", mean_card),
           hjust = -0.08, vjust = 0, size = 2.6, colour = "grey25",
           family = "Helvetica") +
  scale_fill_manual(values = c(`TRUE` = PALETTE_PRIMARY,
                               `FALSE` = alpha(PALETTE_PRIMARY, 0.45)),
                    guide = "none") +
  scale_x_continuous(breaks = card_df$card) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title = "Most images carry three positive labels",
    subtitle = sprintf("Cardinality distribution across %d images (max = %d)",
                       n_imgs, max_card),
    x = "Positive labels per image",
    y = NULL
  ) +
  theme_g3() +
  theme(
    axis.text.y      = element_blank(),
    axis.ticks.y     = element_blank(),
    axis.line.y      = element_blank(),
    panel.grid.major = element_blank()
  )

ggsave(file.path(out_dir, "cardinality.png"), p_card,
       width = 5.2, height = 2.9, dpi = 600, bg = "white")
cat(sprintf("Wrote %s\n", file.path(out_dir, "cardinality.png")))

# ---- 2. long-tail lollipop ------------------------------------------------
# FDLF: don't connect categorical observations with a line — labels have no
# inherent ordering beyond the rank we imposed. A horizontal lollipop sorted
# by frequency shows every label by name (more informative than rank ticks)
# and is honest about the discrete nature of each support count.
ord_idx <- order(-colSums(Y))
counts  <- colSums(Y)[ord_idx]
full    <- vocab[ord_idx]
short   <- short_label(full)
ns      <- namespace_of(full)

# Some short names collide across namespaces (e.g. pattern:blotches vs.
# extra:blotches). Keep the full label as the factor key and only render
# the short form on the axis via scale_y_discrete(labels = ...).
short_unique <- setNames(short, full)

rank_df <- data.frame(
  rank      = seq_along(counts),
  count     = as.integer(counts),
  label     = factor(full, levels = rev(full)),   # tallest at top
  namespace = ns
)

ns_levels  <- sort(unique(rank_df$namespace))
ns_palette <- setNames(
  c(PALETTE_PRIMARY, PALETTE_SECONDARY, PALETTE_ACCENT,
    "#3a7d44", "#d99155")[seq_along(ns_levels)],
  ns_levels
)

singletons <- sum(rank_df$count == 1)
median_n   <- median(rank_df$count)

p_tail <- ggplot(rank_df, aes(y = label, x = count, colour = namespace)) +
  geom_segment(aes(x = 0, xend = count, yend = label), linewidth = 0.45) +
  geom_point(size = 1.9) +
  geom_text(aes(label = count),
            hjust = -0.45, size = 2.4, colour = "grey20",
            family = "Helvetica") +
  scale_colour_manual(values = ns_palette, name = NULL) +
  scale_y_discrete(labels = short_unique) +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0.10)),
    breaks = c(0, 25, 50, 75, 100)
  ) +
  labs(
    title = sprintf("Label support is heavy-tailed (%d singleton labels)",
                    singletons),
    subtitle = sprintf("%d labels, sorted by image count (median = %g)",
                       nrow(rank_df), median_n),
    x = "Images with positive label",
    y = NULL
  ) +
  theme_g3(base_size = 8) +
  theme(
    axis.text.y       = element_text(size = 6.8, colour = "grey20"),
    axis.line.y       = element_blank(),
    axis.ticks.y      = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(colour = PALETTE_GRID,
                                      linewidth = 0.25),
    legend.position   = "top",
    legend.justification = "left",
    legend.key.size   = unit(0.3, "cm"),
    legend.text       = element_text(size = 7),
    legend.margin     = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, -4, -6)
  )

ggsave(file.path(out_dir, "long_tail.png"), p_tail,
       width = 5.6, height = 6.2, dpi = 600, bg = "white")
cat(sprintf("Wrote %s\n", file.path(out_dir, "long_tail.png")))

# ---- 3. co-occurrence heatmap ---------------------------------------------
# FDLF: (a) mask the diagonal — P(i|i) is always 1 and dominates the eye for
# zero information; (b) drop namespace prefixes from tick labels — the legend
# already names the row/column meaning; (c) use viridis (perceptually uniform,
# colorblind-safe) instead of the diverging-feeling YlGnBu ramp.
top_n <- 15
ord   <- order(-colSums(Y))[seq_len(top_n)]
sub   <- Y[, ord, drop = FALSE]
denom <- pmax(colSums(sub), 1)
co    <- (t(sub) %*% sub) / denom

short_ord <- short_label(vocab[ord])
dimnames(co) <- list(short_ord, short_ord)

co_df <- as.data.frame(as.table(co)) |>
  rename(row = Var1, col = Var2, p = Freq) |>
  mutate(
    row        = factor(row, levels = short_ord),
    col        = factor(col, levels = short_ord),
    is_diag    = as.character(row) == as.character(col),
    p_show     = ifelse(is_diag, NA_real_, p),
    txt        = ifelse(!is_diag & p >= 0.10, sprintf("%.2f", p), ""),
    txt_colour = ifelse(!is_diag & p > 0.55, "grey95", "grey15")
  )

p_co <- ggplot(co_df, aes(x = col, y = row)) +
  geom_tile(aes(fill = p_show), colour = "white", linewidth = 0.4) +
  geom_tile(data = subset(co_df, is_diag),
            fill = "grey92", colour = "white", linewidth = 0.4) +
  geom_text(aes(label = txt, colour = txt_colour),
            size = 2.2, family = "Helvetica") +
  scale_fill_viridis_c(
    name    = expression(P(j*"|"*i)),
    option  = "mako",
    direction = -1,
    limits  = c(0, 1),
    breaks  = c(0, 0.25, 0.5, 0.75, 1),
    na.value = "grey92"
  ) +
  scale_colour_identity() +
  scale_y_discrete(limits = rev(short_ord)) +
  labs(
    title    = sprintf("Conditional co-occurrence of the top %d labels",
                       top_n),
    subtitle = expression("Cell value: P(column label " * j *
                          " | row label " * i * "); diagonal omitted"),
    x = NULL, y = NULL
  ) +
  coord_fixed() +
  theme_g3(base_size = 8) +
  theme(
    axis.text.x       = element_text(angle = 45, hjust = 1, size = 7.4),
    axis.text.y       = element_text(size = 7.4),
    panel.grid.major  = element_blank(),
    legend.position   = "right",
    legend.key.height = unit(0.9, "cm"),
    legend.key.width  = unit(0.32, "cm"),
    plot.title        = element_text(size = 9.6, margin = margin(b = 2)),
    plot.subtitle     = element_text(size = 7.6, colour = "grey35",
                                     margin = margin(b = 6))
  )

ggsave(file.path(out_dir, "cooccurrence.png"), p_co,
       width = 6.6, height = 5.6, dpi = 600, bg = "white")
cat(sprintf("Wrote %s\n", file.path(out_dir, "cooccurrence.png")))
