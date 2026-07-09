import { rustModule } from '../generated/rust.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(rustModule);

export { rustModule };
