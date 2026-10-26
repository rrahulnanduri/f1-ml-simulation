// Plain-language content for every figure, written for someone who has never followed motor racing.
// Merged into each figure by build.mjs: the intro box, the parts list, the glossary, the wording of
// the walk-through steps and the one-line meaning under every number.
// Step entries replace only the title and text of the step at the same position.

export const EXPLAIN = {
  chassis: {
    plain: {
      idea: [
        "A Formula 1 car is built around a very strong, very light carbon-fibre tub that the driver sits inside. Everything else bolts onto it.",
        "If the car hits a wall, the cone at the front crumples like a drinks can and soaks up the crash energy, while the tub stays whole. That is how drivers walk away from crashes that look terrifying.",
      ],
      tryFirst: [
        "Press Run the test.| Watch the dark cone at the front squash while the cell around the driver stays intact.",
        "Switch the test to Halo.| The Halo is the curved bar over the driver's head. Raise the load and see how little it bends.",
        "Turn the carbon skin down.| You can then see the driver, seat, fuel tank and electronics inside the tub.",
      ],
    },
    parts: [
      { name: "The tub (monocoque)", groups: ["shell"], text: "The strong carbon-fibre shell around the driver, also called the survival cell. It is made to stay rigid in a crash." },
      { name: "The Halo", groups: ["halo"], text: "A curved titanium bar over the driver's head, held on three mounts. It stops wheels and debris reaching the head." },
      { name: "Driver, seat and harness", groups: ["driver"], text: "The driver sits low and leaning back, with feet ahead of the front wheels, strapped in by a six-point harness." },
      { name: "Fuel tank", groups: ["fuel"], text: "A flexible rubber bag behind the seat, inside the strongest part of the car so it is not punctured in a crash." },
      { name: "Battery and electronics", groups: ["electronics"], text: "The energy store and the control box, kept inside the tub where they are protected." },
      { name: "Crash cone at the front", groups: ["crash"], text: "A hollow carbon cone made to be crushed. It takes the energy of a head-on crash so the driver does not." },
      { name: "Side crash structures", groups: ["side"], text: "Crushable carbon panels along the side of the tub that soak up a hit from the side." },
      { name: "The test rig and graph", groups: ["rig", "plot"], text: "A sled that runs into a wall, and a graph of how hard the driver is slowed down. The bar across the graph is the limit." },
    ],
    glossary: [
      { term: "Monocoque", text: "French for 'single shell'. Here: one moulded carbon-fibre tub that is both the body and the safety cell." },
      { term: "Carbon fibre", text: "Thin woven threads of carbon set in resin. It is far lighter than steel and, in the right layout, just as strong." },
      { term: "g (g-force)", text: "A push measured in multiples of your own weight. 20 g means the driver feels twenty times heavier for an instant." },
      { term: "m/s", text: "Metres per second. 15 m/s is 54 km/h, the speed of the official frontal crash test." },
      { term: "kJ (kilojoule)", text: "A unit of energy. The test crash carries about the energy needed to lift a one-tonne car nine metres." },
      { term: "kN (kilonewton)", text: "A unit of force. 100 kN is roughly the weight of ten tonnes." },
    ],
    stats: {
      peak: "How hard the driver is slowed at the worst instant. The limit in the rules is 60.",
      avg: "The slowing averaged over the whole crash. The target is about 25 or less.",
      crush: "How far the front of the car squashes. More distance means a gentler stop.",
      energy: "How much crash energy the structure turns into bent carbon.",
      intrusion: "How far anything pushes into the driver's space. It should stay at zero.",
      head: "How far the driver's head moves forward inside the cell.",
      halo: "Force pushed on the Halo, and how far it bends.",
      result: "Whether the structure passed or failed the test.",
    },
    steps: [
      { title: "A carbon shell around the driver", text: "The tub is one moulded carbon-fibre shell, light enough to lift by hand and strong enough to survive a crash. Everything else on the car bolts to it." },
      { title: "The driver sits low inside it", text: "Fade the skin and you can see the driver sitting low, leaning back, with feet ahead of the front wheels, held in by a six-point harness and a neck support." },
      { title: "Fuel and electronics are inside too", text: "The fuel bag sits behind the seat and the battery and control box sit in front of the legs, all inside the strongest part of the car." },
      { title: "The Halo", text: "A titanium bar on three mounts above the driver's head. It is tested with a force of roughly ten tonnes and bends only a few millimetres." },
      { title: "Crush the nose, not the cell", text: "In the crash test the front cone crumples from the tip backwards, soaking up the energy over about half a metre. The cell, and the driver, stop gently." },
      { title: "Not too soft, not too stiff", text: "A cone that is too soft runs out of room and the cell is hit. A cone that is too stiff slows the driver too sharply. The design has to land in between." },
      { title: "The sides are protected too", text: "Crushable panels along the sides take a hit from a ram. Only if they are overwhelmed does the wall of the cell itself start to dent." },
    ],
  },

  "power-unit": {
    plain: {
      idea: [
        "An F1 engine is a small petrol engine with a turbocharger, plus two electric motors and a battery. Together they are called the power unit.",
        "The electric parts catch energy that a normal car throws away, the heat of the exhaust and the braking, store it in the battery and feed it back as extra push. That is how the car reaches over 1,000 horsepower from a 1.6-litre engine.",
      ],
      tryFirst: [
        "Drag left and right across the drawing.| That revs the engine. Watch the pistons move and the glow jump from cylinder to cylinder.",
        "Set the energy mode to deploy.| The battery adds power and the total passes 1,000 horsepower.",
        "Set the mode to harvest and press the brake.| Green dashes show energy flowing back into the battery.",
      ],
    },
    parts: [
      { name: "The V6 engine", groups: ["ice"], text: "Six cylinders in two rows of three. Each fuel explosion pushes a piston down and turns the crankshaft. It spins up to 15,000 times a minute." },
      { name: "Turbocharger", groups: ["turbo"], text: "A fan driven by the hot exhaust that spins a second fan to pack more air into the engine, which means more power." },
      { name: "MGU-H (heat motor)", groups: ["mguh"], text: "A small electric motor-generator on the turbo's shaft. It turns surplus exhaust heat into electricity, or spins the turbo up to remove lag." },
      { name: "MGU-K (motion motor)", groups: ["mguk"], text: "An electric motor-generator on the engine's crankshaft. It adds extra push when accelerating and recovers energy when braking." },
      { name: "Battery (energy store)", groups: ["es"], text: "Stores the recovered electricity until it is needed. Only a limited amount may be used each lap." },
      { name: "Control electronics", groups: ["ce"], text: "The computer and power electronics that decide when each motor drives or recovers." },
      { name: "Air path", groups: ["intake"], text: "Air is drawn in, squeezed by the turbo, cooled in the radiator-like box, and fed to the cylinders." },
      { name: "Exhaust", groups: ["exhaust"], text: "Hot gas leaves each cylinder, spins the turbine, then goes out of the tailpipe." },
      { name: "Flow lines", groups: ["flows"], text: "Red dashes are exhaust gas, blue is air, green is electricity. They move faster as the engine speeds up." },
    ],
    glossary: [
      { term: "rpm", text: "Revolutions per minute: how many times the engine's crankshaft turns in a minute. A road car rarely passes 6,000; an F1 engine reaches 15,000." },
      { term: "Horsepower and kW", text: "Two units of power. 1 kW is about 1.34 horsepower. 745 kW is 1,000 horsepower." },
      { term: "Turbocharger", text: "Uses exhaust gas to spin a fan that forces extra air into the engine." },
      { term: "Hybrid", text: "Using both a fuel engine and electric motors." },
      { term: "Deploy and harvest", text: "Deploy means using stored electricity to add power. Harvest means making electricity, from braking or exhaust heat, and storing it." },
      { term: "MJ (megajoule)", text: "A unit of energy, a million joules. The rules let the car use 4 MJ from the battery per lap and recover 2 MJ by braking." },
      { term: "Boost", text: "How hard the turbo squeezes the incoming air. 1 bar is normal air pressure." },
    ],
    stats: {
      rpm: "How fast the engine is spinning.",
      ice: "Power from the petrol engine alone.",
      mguk: "Power added (+) or taken away to recharge (-) by the motion motor.",
      mguh: "Power the heat motor takes from the exhaust (+) or uses to spin the turbo (-).",
      total: "All power reaching the wheels, in kilowatts and horsepower.",
      boost: "How much the turbo squeezes the air.",
      soc: "How full the battery is.",
      fuel: "Fuel burned per hour. The rules cap it at 100 kg an hour.",
      deploy: "Energy taken out of the battery this lap, out of the 4 MJ allowed.",
      harvest: "Energy put back into the battery this lap, out of the 2 MJ allowed.",
    },
    steps: [
      { title: "A small V6 with a turbo", text: "Six cylinders in two rows of three. Watch the glow move from cylinder to cylinder: each one fires in turn to keep the engine turning." },
      { title: "Exhaust heat becomes power", text: "Hot exhaust spins a fan. That fan drives the compressor that packs air in, and any extra turning power is turned into electricity by the heat motor." },
      { title: "The motion motor adds push", text: "With the throttle down in deploy mode, the electric motor on the crankshaft adds up to 120 kW straight to the engine, fed by the battery and the heat motor." },
      { title: "Braking charges the battery", text: "When the driver brakes, the same motor works backwards as a generator and recovers energy. The green dashes now run into the battery." },
      { title: "Over 1,000 horsepower", text: "At full throttle the engine alone makes about 640 kW. Add 120 kW from the electric motor and the total passes 1,000 horsepower." },
      { title: "A budget for each lap", text: "Only 4 MJ may leave the battery in a lap and only 2 MJ may be recovered. Use it all in one go and the extra push stops until you start a new lap." },
    ],
  },

  aero: {
    plain: {
      idea: [
        "Aerodynamics is how air flows around the car. The wings work like upside-down aeroplane wings: instead of lifting, they push the car down onto the road. More push down means more grip, so the car can take corners much faster.",
        "The price is drag, the air pushing back. DRS is a flap that opens on straights to let air through, cutting drag so the car goes faster.",
      ],
      tryFirst: [
        "Raise the tunnel speed.| The push-down force grows very quickly with speed.",
        "Click the rear wing, or press Space.| The flap opens: drag drops and the top speed rises.",
        "Lower the ride height.| The floor works harder, until the car is too low and the air stops flowing properly.",
      ],
    },
    parts: [
      { name: "Front wing", groups: ["frontwing"], text: "The first part to meet the air. It pushes the front down and steers the air around the front wheels towards the rest of the car." },
      { name: "The floor and its tunnels", groups: ["floor"], text: "Under the car, shaped channels squeeze the air so it speeds up and its pressure drops. The car is sucked down onto the road. It makes nearly half the downforce." },
      { name: "Diffuser", groups: ["diffuser"], text: "The curved ramp at the back of the floor. It lets the fast air slow down gently, which keeps the suction under the car strong." },
      { name: "Rear wing and DRS flap", groups: ["rearwing"], text: "Pushes the back down. The top flap can open on a straight to cut drag. That opening is called DRS." },
      { name: "Wheels", groups: ["wheels"], text: "Spinning wheels churn the air into a mess, so the wings and bodywork are shaped to steer air around them." },
      { name: "Airflow lines", groups: ["flow"], text: "Dashed lines show where the air goes. They turn blue and bunch up where the air is fastest, under the floor." },
      { name: "Force arrows", groups: ["forces"], text: "Black arrows are the push down. The red arrow is drag, pushing the car backwards." },
      { name: "Ride height gauge", groups: ["gauge"], text: "Measures how high the floor sits above the road. A few centimetres decides how well the floor works." },
      { name: "Car body", groups: ["body"], text: "Nose, cockpit, sidepods and engine cover, all shaped to guide the air." },
      { name: "Wind tunnel", groups: ["rig"], text: "A moving belt for a road and a rake of smoke pipes at the front that make the air visible." },
    ],
    glossary: [
      { term: "Downforce", text: "Air pushing the car down onto the road. It adds grip without adding weight." },
      { term: "Drag", text: "Air pushing back against the car. It limits the top speed." },
      { term: "DRS", text: "Drag Reduction System. The driver opens a flap on the rear wing on straights so less air resists the car." },
      { term: "Ride height", text: "How far the floor of the car is above the track." },
      { term: "Stall", text: "When the air under the floor stops flowing smoothly, the suction collapses and downforce drops sharply." },
      { term: "Porpoising", text: "The car bounces up and down at speed as the floor alternately stalls and recovers." },
      { term: "kN (kilonewton)", text: "A unit of force. 10 kN is about a tonne pushing down." },
    ],
    stats: {
      down: "Total push down onto the road.",
      split: "How the push down is shared between front and rear.",
      drag: "Force of the air pushing back.",
      ld: "Push down gained for each unit of drag. Higher is more efficient.",
      floor: "Share of downforce made by the floor and diffuser.",
      vmax: "Rough top speed for a fixed engine power against this drag.",
      gain: "How much faster the top speed becomes with DRS open.",
    },
    steps: [
      { title: "The front wing meets the air first", text: "It makes about a fifth of the push down and decides how the air reaches the rest of the car. A steeper flap pushes harder, and drags more." },
      { title: "The floor is the biggest wing", text: "Fade the car and look underneath. Air squeezed through the tunnels speeds up and its pressure drops, so the car is sucked onto the road." },
      { title: "The diffuser lets the air go", text: "The rising ramp at the back slows the fast air down gently, which keeps the suction under the floor strong all the way along." },
      { title: "The rear wing balances the front", text: "A steeper rear wing pushes the back down harder, and adds a lot more drag. Teams choose the angle for each track." },
      { title: "DRS on a straight", text: "The driver opens the rear flap. Air flows through the gap, drag falls by about a fifth and the top speed rises. It closes again when braking." },
      { title: "Too low and the floor stalls", text: "Lower the car and the floor works harder, until the gap is too small and the air stops flowing. The push down collapses, the car rises, and then it drops again: a bounce called porpoising." },
    ],
  },

  drivetrain: {
    plain: {
      idea: [
        "The gearbox passes the engine's power to the wheels. Low gears give strong pull for pulling away; high gears give high speed. An F1 car has eight, and the driver changes them with two paddles behind the steering wheel.",
        "The steering wheel is also the control panel. Its dials change how the brakes and the engine behave, and it pops off so the driver can climb in and out.",
      ],
      tryFirst: [
        "Press the upshift button.| Watch the highlighted gear change as the little drum at the back turns and slides a ring.",
        "Turn Shift for me off and use the up and down arrow keys.| The numbers on the wheel's screen change with you.",
        "Press Take the wheel off.| The screen goes blank and the paddles stop working.",
      ],
    },
    parts: [
      { name: "Steering wheel", groups: ["wheel"], text: "A small control panel with a screen, two paddles behind it and rotary switches. Click the dials to turn them, click the paddles to change gear." },
      { name: "Upper shaft and its gears", groups: ["main"], text: "The engine turns this shaft. The eight gears on it are always turning with it." },
      { name: "Lower shaft and its gears", groups: ["lay"], text: "Each gear here meshes with one above. Only one at a time is locked to the shaft, the highlighted one, so that is the gear you are in." },
      { name: "Shift drum, forks and rings", groups: ["shift"], text: "A drum with curved grooves turns a little for each gear change. Its grooves slide a fork, which slides a ring that locks the next gear to the shaft." },
      { name: "Gearbox casing", groups: ["case"], text: "The housing around the gears, cut away here so you can see inside. The big disc at the back passes the power on to the wheels." },
    ],
    glossary: [
      { term: "Gear ratio", text: "How many times the engine turns for one turn of the wheels. A big ratio gives strong pull, a small one high speed." },
      { term: "Sequential", text: "You can only go up or down one gear at a time, in order, with no skipping." },
      { term: "Paddle shifter", text: "A lever behind the steering wheel that the driver pulls to change gear." },
      { term: "Dog ring", text: "A ring that slides along a shaft and locks a free-spinning gear to it with interlocking teeth." },
      { term: "Brake balance", text: "How the braking force is split between the front and rear wheels. The driver adjusts it with a dial." },
      { term: "Differential", text: "Lets the two driven wheels turn at different speeds in a corner. Its setting changes how the car handles when accelerating out of corners." },
      { term: "Engine mode", text: "A setting that changes how much power the engine and the electric motors give." },
      { term: "rpm", text: "How many times the engine turns each minute." },
    ],
    stats: {
      speed: "How fast the car is moving.",
      rpm: "How fast the engine is spinning.",
      gear: "Which of the eight gears is selected.",
      ratio: "Engine turns for each wheel turn in this gear.",
      torque: "The twisting force the engine gives.",
      power: "Engine power right now.",
      accel: "How hard the car is accelerating, in multiples of gravity.",
      split: "Brake force split between front and rear wheels, and the force on the front.",
    },
    steps: [
      { title: "Eight pairs of gears", text: "The engine turns the upper shaft and every gear on it. Each one meshes with a gear on the lower shaft that spins freely until a ring locks one pair. The big pair is first gear; the nearly equal pair is eighth." },
      { title: "A paddle pull turns a drum", text: "One pull turns the drum a little. Its grooves slide a fork, the fork slides a ring, and the ring locks the next gear. In the real car this takes about 28 thousandths of a second." },
      { title: "Why eight gears", text: "Each gear keeps the engine in its best range over a different slice of road speed. First tops out near 85 km/h, eighth near 350 km/h." },
      { title: "The wheel is the control room", text: "The dials set brake balance, the differential and the engine mode, and the screen shows the gear, speed and rev lights. Turn them here or click them on the wheel." },
      { title: "Brake balance", text: "More brake to the front stops the car harder but risks locking the front tyres. More to the rear is steadier until the rears lock. The driver trims it corner by corner." },
      { title: "Take the wheel off", text: "The wheel clips off so the driver can get in and out. With it off, the car cannot be shifted: the screen goes blank and the paddles do nothing." },
    ],
  },

  corner: {
    plain: {
      idea: [
        "Each wheel hangs from the car on a set of arms, the suspension, that let it move up and down over bumps. A spring and a damper stop it bouncing. A brake disc inside the wheel turns speed into heat. The tyre is the only part that touches the road.",
        "This is one front wheel on a test rig, with the road shown as a pad under the tyre that you can push.",
      ],
      tryFirst: [
        "Press Hit the kerb.| The wheel jumps, the rod and the rocker turn, and the damper soaks up the shock.",
        "Raise Speed to 300 km/h.| Air pushing down on the car squeezes the suspension.",
        "Turn on See through the tyre, then hold the brake.| The brake disc glows as it turns speed into heat.",
      ],
    },
    parts: [
      { name: "Tyre", groups: ["tyre"], text: "The only part that touches the road. The patch of rubber on the road is about the size of a postcard." },
      { name: "Brake disc and caliper", groups: ["brake"], text: "A carbon disc turns with the wheel. The caliper squeezes pads against it and the friction slows the car. The disc glows hot." },
      { name: "Wishbone arms and upright", groups: ["suspension"], text: "Two pairs of V-shaped arms join the wheel's carrier to the car and let it move up and down in a controlled way." },
      { name: "Push rod", groups: ["pushrod"], text: "A strut that pushes on the rocker when the wheel moves up. The red-outlined one is the rod that carries the bump." },
      { name: "Rocker, torsion bar and damper", groups: ["torsion", "damper"], text: "The rocker turns a twisting bar that acts as a spring and squeezes the damper, which soaks up energy so the car does not keep bouncing." },
      { name: "Car chassis stub", groups: ["chassis"], text: "The part of the car's body that the suspension is attached to. It is allowed to move up and down on guide rods." },
      { name: "Road pad and rams", groups: ["pad", "rig"], text: "The pad is the road. The rams under it push it up and down to make a bump." },
      { name: "Grip circle", groups: ["grip"], text: "Shows how much of the tyre's grip is being used. The dot at the edge means the tyre is at its limit." },
    ],
    glossary: [
      { term: "Suspension", text: "The arms, spring and damper that connect the wheel to the car and let it move over bumps." },
      { term: "Damper", text: "A piston in oil that resists movement. It stops the spring bouncing on and on." },
      { term: "Torsion bar", text: "A steel bar that acts as a spring by twisting." },
      { term: "Kerb", text: "The raised stripe at the edge of a corner that drivers cut across." },
      { term: "Downforce", text: "Air pushing the car down onto the road as it goes faster." },
      { term: "Grip", text: "How well the tyre holds onto the road. It has to be shared between braking and turning." },
      { term: "Caliper", text: "The clamp that squeezes brake pads onto the disc." },
      { term: "Contact patch", text: "The small area of tyre that is touching the road at any moment." },
    ],
    stats: {
      speed: "How fast the car is going.",
      load: "How hard the tyre is pressed onto the road.",
      grip: "How much of the tyre's available grip is being used. Over 100% means it slides.",
      travel: "How far the suspension has been squeezed.",
      tyre: "How far the tyre is squashed, and the length of the patch touching the road.",
      disc: "Temperature of the brake disc.",
      decel: "How hard the car is slowing, in multiples of gravity.",
    },
    steps: [
      { title: "The tyre is the only contact", text: "Everything the car does to the road goes through four patches of rubber, each about the size of a postcard. Standing still, this one carries about 180 kg." },
      { title: "Speed squeezes the suspension", text: "At 300 km/h the wings push about 330 kg more onto this wheel. The suspension is squeezed, the car sits lower, and the tyre is squashed more." },
      { title: "How the suspension takes a bump", text: "A kerb throws the wheel up. The push rod turns the rocker, which squeezes the damper to soak up the shock and twists the bar like a spring." },
      { title: "Brakes turn speed into heat", text: "Hard braking from 300 km/h pushes about a megawatt of heat into one front disc. It glows and climbs towards 1,000 °C, then cools in the airstream. The tyre is faded so you can see inside the wheel." },
      { title: "The grip circle", text: "A tyre has one budget of grip to share between braking and turning. Ask for both and the dot reaches the edge of the circle: the tyre is at its limit." },
    ],
  },
};

/** What the whole car is, in plain words, for the index page. */
export const OVERVIEW = {
  title: "How a Formula 1 car works",
  lead: "A Formula 1 car is the fastest road-racing machine on Earth: it can pass 350 km/h, and it can corner with more than five times its own weight pressing it into the track. You do not need to know anything about racing to follow this. Each of the five pages explains one system of the car, in plain words, with a model you can play with.",
  how: [
    { name: "1. The tub keeps the driver safe", text: "A carbon shell around the driver, with a crumple cone at the front." },
    { name: "2. The power unit makes the push", text: "A small turbo petrol engine helped by electric motors." },
    { name: "3. The air pushes the car down", text: "Wings and the underside make grip out of the air itself." },
    { name: "4. The gearbox and wheel are the controls", text: "Eight gears and a steering wheel that is a mini control room." },
    { name: "5. The wheels, springs and brakes touch the road", text: "Suspension, brakes and tyres turn all of that into speed and stopping." },
  ],
};
