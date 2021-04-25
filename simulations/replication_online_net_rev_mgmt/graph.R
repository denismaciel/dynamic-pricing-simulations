library(tidyverse)

df <- list.files('data', pattern = '*.csv', full.names = TRUE) %>%
  map(function(x) {
    df <- read_csv(x)
    df$experiment <- x
    return(df)
  }) %>%
  bind_rows

CLAIRVOYANT_REVENUE <- (0.75 * 0.3 * 39.9) + (0.25 * 0.1 * 44.9)

# Sanity check
df %>%
  group_by(experiment, id) %>%
  summarise(n_periods = max(step) + 1, n = n()) %>%
  group_by(experiment) %>%
  summarise(mean(n_periods))


# Cleaner approach
trial_names <- tibble(
  trial_factory = c(
    "data/ts_fixed_with_bernoulli_.csv",
    "data/ts_update_with_bernoulli_.csv",
    "data/ts_ignore_inventory_with_bernoulli_.csv"
  ),
  algorithm = c("TS-Fixed", "TS-Update", "TS")
)

avg_revenue <- df %>%
  group_by(experiment, trial_id = id) %>%
  summarise(total_revenue = sum(price * demand, na.rm = TRUE)) %>%
  mutate(
    n_periods = as.integer(str_extract(experiment, "\\d+")),
    trial_factory = str_remove(experiment, as.character(n_periods))
  ) %>%
  left_join(trial_names) %>%
  mutate(avg_revenue_per_step = total_revenue / n_periods) %>%
  group_by(algorithm, n_periods) %>%
  summarise(as_pp_of_clairvoyant = mean(avg_revenue_per_step,  na.rm = TRUE) / CLAIRVOYANT_REVENUE)

x_axis <-  tibble(labels = unique(avg_revenue$n_periods)) %>%
  mutate(breaks = log(labels))

avg_revenue %>%
  ggplot(aes(
    log(n_periods),
    as_pp_of_clairvoyant,
    color = algorithm,
    shape = algorithm
  )) +
  geom_point() +
  geom_line() +
  labs(x = "Number of Periods T (in log scale)",
       y = NULL,
       title = "Percent of Optimal Revenue Achieved") +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent,
                     breaks = scales::pretty_breaks(n = 8)) +
  scale_x_continuous(labels = x_axis$labels, breaks = x_axis$breaks) +
  theme(
    legend.title = element_blank(),
    legend.position = "bottom",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
  )
