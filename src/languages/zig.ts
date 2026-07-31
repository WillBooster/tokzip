import { zigModule } from '../generated/zig.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(zigModule);

export { zigModule };
