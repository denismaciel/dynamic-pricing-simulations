library(tidyverse)

ggplot2::theme_set(theme_light())

FIGS_FOLDER <- fs::path("figs")

df <- list.files('simulations/competition/data', pattern = '*.csv', full.names = TRUE) %>%
  map(
    function(x) 
      read_csv(x) %>%
      mutate(experiment = fs::path_file(x) %>% stringr::str_remove(".csv"))
  ) %>%
  bind_rows

df %>% 
  count(experiment,firm, period_id, price) %>% 
  group_by(firm, period_id) %>% 
  mutate(pp_chosen = n / sum(n)) %>% 
  ggplot(aes(period_id, pp_chosen, color = factor(price))) +
  geom_line() +
  facet_grid(firm ~ .) +
  labs(color = "Price", x = "Period", y = NULL) +
  theme(legend.position = 'bottom')

avg_revenue_per_period <- df %>% 
  group_by(experiment, firm, period_id) %>% 
  summarise(revenue = mean(revenue)) %>% 
  ggplot(aes(period_id, revenue, color=firm)) +
  geom_line(alpha=0.5) +
  facet_wrap(~experiment) + 
  labs(
    title='', 
    x = NULL,
    y = "Average Revenue",
    color = "Firm"
  ) +
  theme(legend.position = 'bottom')

ggsave(fs::path('figs', 'competition_avg_revenue_per_period.png'), avg_revenue_per_period)


# Not used
df %>% 
  filter(trial_id == 5) %>% 
  ggplot(aes(period_id, revenue, color=factor(price))) +
  geom_point(alpha=1/5) +
  facet_wrap(. ~ firm) +
  labs(x = NULL, y = "Revenue", color = "Firm")

revenue_per_price_combo <- df %>% 
  filter(experiment == "ts_fixed_vs_itself") %>% 
  pivot_wider(id_cols = c(trial_id, period_id), 
              names_from = firm, 
              names_sep = "_", 
              values_from = c(demand, price, revenue)) %>% 
  group_by(price_A = factor(price_A), price_B = factor(price_B)) %>% 
  summarise(mean_revenue_A = mean(revenue_A)) %>%
  ungroup() %>% 
  mutate(mean_revenue_A = mean_revenue_A / mean(mean_revenue_A)) %>%
  ggplot(aes(price_A, price_B)) +
  geom_raster(aes(fill=mean_revenue_A)) +
  geom_text(aes(price_A, price_B, label = round(mean_revenue_A, 3)), color="white") +
  labs(x = "Price A", y = "Price B", fill = "Mean Revenue") +
  theme_minimal()

ggsave(fs::path('figs', 'competition_revenue_per_price_combo.png'), revenue_per_price_combo)
