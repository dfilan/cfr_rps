// Code to solve Rock Paper Scissors using Counterfactual Regret Minimization
// Following "An Introduction to Counterfactual Regret Minimization" by Neller and Lanctot (2013)

use rand::Rng;

// TODO: rewrite to get equilibrium

fn main() {
    let num_iters = 1_000_000; // TODO: should be read from command line
    let mut rng = rand::thread_rng();
    let opp_strategy = ThrowVals {
        rock: 0.4,
        paper: 0.3,
        scissors: 0.3,
    };

    let mut strategy_sum = new_vals();
    let mut regret_sum = new_vals();

    train_strategy(
        num_iters,
        &mut strategy_sum,
        &mut regret_sum,
        &opp_strategy,
        &mut rng,
    );

    let avg_strategy = get_average_strategy(&strategy_sum);

    println!("Average strategy is {:?}", avg_strategy);
}

#[derive(Copy, Clone)]
enum RpsThrow {
    Rock,
    Paper,
    Scissors,
}
const THROW_LIST: [RpsThrow; 3] = [RpsThrow::Rock, RpsThrow::Paper, RpsThrow::Scissors];

fn game_value(my_throw: &RpsThrow, opp_throw: &RpsThrow) -> f64 {
    match (my_throw, opp_throw) {
        (RpsThrow::Rock, RpsThrow::Rock) => 0.0,
        (RpsThrow::Paper, RpsThrow::Paper) => 0.0,
        (RpsThrow::Scissors, RpsThrow::Scissors) => 0.0,
        (RpsThrow::Rock, RpsThrow::Paper) => -1.0,
        (RpsThrow::Paper, RpsThrow::Scissors) => -1.0,
        (RpsThrow::Scissors, RpsThrow::Rock) => -1.0,
        _other => 1.0,
    }
}

#[derive(Copy, Clone, Debug)]
struct ThrowVals {
    rock: f64,
    paper: f64,
    scissors: f64,
}

impl ThrowVals {
    fn get_val(&self, throw: &RpsThrow) -> f64 {
        match throw {
            RpsThrow::Rock => self.rock,
            RpsThrow::Paper => self.paper,
            RpsThrow::Scissors => self.scissors,
        }
    }

    // change this to return a mutable reference?
    fn write_val(&mut self, throw: &RpsThrow) -> &mut f64 {
        match throw {
            RpsThrow::Rock => &mut self.rock,
            RpsThrow::Paper => &mut self.paper,
            RpsThrow::Scissors => &mut self.scissors,
        }
    }
}

fn new_vals() -> ThrowVals {
    ThrowVals {
        rock: 0.0,
        paper: 0.0,
        scissors: 0.0,
    }
}

fn get_strategy(regret_sum: &ThrowVals, strategy_sum: &mut ThrowVals) -> ThrowVals {
    let mut strategy = ThrowVals {
        rock: 0.0,
        paper: 0.0,
        scissors: 0.0,
    };
    let mut normalizing_sum = 0.0;

    for throw in THROW_LIST {
        let r = regret_sum.get_val(&throw);
        let throw_strat = if r > 0.0 { r } else { 0.0 };
        *strategy.write_val(&throw) = throw_strat;
        normalizing_sum += throw_strat;
    }

    for throw in THROW_LIST {
        let throw_strat = strategy.get_val(&throw);
        let new_strat = if normalizing_sum > 0.0 {
            throw_strat / normalizing_sum
        } else {
            1.0 / (THROW_LIST.len() as f64)
        };
        *strategy.write_val(&throw) = new_strat;
        *strategy_sum.write_val(&throw) += new_strat;
    }

    strategy
}

fn sample_strategy(strategy: &ThrowVals, rng: &mut rand::rngs::ThreadRng) -> RpsThrow {
    // TODO: somehow check that weights approximately add to one...
    let r: f64 = rng.gen();
    let mut cum_prob = 0.0;
    for throw in THROW_LIST {
        cum_prob += strategy.get_val(&throw);
        if cum_prob > r {
            return throw;
        }
    }
    THROW_LIST[THROW_LIST.len() - 1]
}

fn get_action_pair(
    regret_sum: &ThrowVals,
    strategy_sum: &mut ThrowVals,
    opp_strategy: &ThrowVals,
    rng: &mut rand::rngs::ThreadRng,
) -> (RpsThrow, RpsThrow) {
    let my_strategy = get_strategy(regret_sum, strategy_sum);
    let my_action = sample_strategy(&my_strategy, rng);
    let opp_action = sample_strategy(&opp_strategy, rng);
    (my_action, opp_action)
}

fn my_utilities(opp_action: &RpsThrow) -> ThrowVals {
    let mut utilities = new_vals();
    for throw in THROW_LIST {
        *utilities.write_val(&throw) = game_value(&throw, opp_action);
    }
    utilities
}

fn accumulate_regrets(my_action: &RpsThrow, my_utilities: &ThrowVals, regret_sum: &mut ThrowVals) {
    for throw in THROW_LIST {
        *regret_sum.write_val(&throw) +=
            my_utilities.get_val(&throw) - my_utilities.get_val(my_action);
    }
}

fn train_strategy(
    num_iters: i32,
    strategy_sum: &mut ThrowVals,
    regret_sum: &mut ThrowVals,
    opp_strategy: &ThrowVals,
    rng: &mut rand::rngs::ThreadRng,
) {
    for _i in 0..num_iters {
        let (my_action, opp_action) = get_action_pair(regret_sum, strategy_sum, opp_strategy, rng);
        let my_utils = my_utilities(&opp_action);
        accumulate_regrets(&my_action, &my_utils, regret_sum);
    }
}

fn get_average_strategy(strategy_sum: &ThrowVals) -> ThrowVals {
    let mut avg_strat = new_vals();
    let mut normalizing_sum = 0.0;
    for throw in THROW_LIST {
        normalizing_sum += strategy_sum.get_val(&throw);
    }
    for throw in THROW_LIST {
        *avg_strat.write_val(&throw) = if normalizing_sum > 0.0 {
            strategy_sum.get_val(&throw) / normalizing_sum
        } else {
            1.0 / (THROW_LIST.len() as f64)
        };
    }
    avg_strat
}
